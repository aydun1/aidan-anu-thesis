#!/usr/bin/env python
import requests
import json
import statistics
import time
import re

from pathlib import Path

gt_url = 'https://gt-scan.csiro.au'
ens_url = 'https://rest.ensembl.org'
genome_path = '/scratch1/obr17q/GRCh38'
sequence_path = './sequences'
gtscan_params = {'mismatches': 3, 'genome': 'GRCh38', 'mode': 'summary'}

cas_params = {
    'cas9': {
        'length': 23,
        'forward_re': re.compile('^([ACGT]{20})[ACGT]GG$', re.IGNORECASE),
        'reverse_re': re.compile('^CC[ACGT]([ACGT]{20})$', re.IGNORECASE),
        'gtscan_rule': 'xxxxxxxxxxXXXXXXXXXXNGG',
        'gtscan_filter': 'NNNNNNNNNNNNNNNNNNNNNGG',
        'casoffinder_rule': 'NNNNNNNNNNNNNNNNNNNNNGG'
    },
    'cas12a': {
        'length': 24,
        'forward_re': re.compile('^TTT[ACGT]([ACGT]{20})$', re.IGNORECASE),
        'reverse_re': re.compile('^([ACGT]{20})[ACGT]AAA$', re.IGNORECASE),
        'gtscan_rule': 'TTTNXXXXXXXXXXxxxxxxxxxx',
        'gtscan_filter': 'TTTNNNNNNNNNNNNNNNNNNNNN',
        'casoffinder_rule': 'TTTNNNNNNNNNNNNNNNNNNNNN'
    }
}

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)


def gc_content(seq):
    d = len([s for s in seq if s in 'CcGc']) / len(seq) * 100
    return round(d, 2)


def reverse_complement(sequence):
    defs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(reversed([defs[n] if n in defs else n for n in sequence]))


def parse_fasta(gene):
    with open(f'./sequences/Homo_sapiens_{gene}_sequence.fa') as f:
        header = f.readline().rstrip().split(':')
        seq = ''.join([line.rstrip() for line in f])
        strand = int(header[6])
        seq = seq if strand == 1 else reverse_complement(seq)
        fasta = {'chromosome': header[3], 'strand': strand, 'position': int(header[4]), 'sequence': seq}
        return fasta


def find_guides(fasta, effector):
    guide_length = cas_params[effector]['length']
    guides = []
    for start in range(len(fasta['sequence'])):
        window = fasta['sequence'][start:start + guide_length]
        sequence = ''
        if cas_params[effector]['forward_re'].match(window):
            sequence = window
            spacer = cas_params[effector]['forward_re'].match(window).group(1)
            strand = '+'
        elif cas_params[effector]['reverse_re'].match(window):
            sequence = reverse_complement(window)
            spacer = reverse_complement(cas_params[effector]['reverse_re'].match(window).group(1))
            strand = '-'
        if sequence:
            guides.append([sequence, start + fasta['position'], strand, spacer, fasta['chromosome']])
    return guides

 
def filter_exonic_guides(guides, gene):
    with open('genes.json') as f:
        coords = json.load(f)['genes']
    gene_info = next(c for c in coords if c['name'] == gene)
    exons = [[min(e['start'], e['end']), max(e['start'], e['end']) - 1] for e in gene_info['exons']]

    lengths = [e[1] - e[0] for e in exons]
    print(len(lengths), 'exons with an average length of', sum(lengths) / len(lengths), 'and total of', sum(lengths))

    overlap = 10 #ensure guide is at least 10 bases into exon
    exonic_guides = []
    for g in guides:
        intersects = [g[1] + overlap <= e[1] and g[1] + len(g[0]) - overlap >= e[0] for e in exons]
        if sum(intersects):
            exon_number = intersects.index(True) + 1
            g.append(exon_number)
            exonic_guides.append(g)
    overlaps = [[g[1] + overlap <= e[1] and g[1] + len(g[0]) - overlap >= e[0] for e in exons] for g in guides]
    res = [sum(i) for i in zip(*overlaps)] 
    print(res)
    guide_count = [len(g) for g in exonic_guides]
    return exonic_guides


def save_casoffinder_input(guides, effector):
    if effector == 'cas9':
        queries = [f'{g[0][:-3]}NNN 5' for g in guides]
    elif effector == 'cas12a':
        queries = [f'NNNN{g[0][4:]} 5' for g in guides]
    with open(f'./guides/guides_{gene}_{effector}.txt', 'w') as f:
        f.write(genome_path + '\n')
        f.write(cas_params[effector]['casoffinder_rule'] + '\n')
        f.writelines(f'{line}\n' for line in queries)

    print(f'./cas-offinder ./guides/guides_{gene}_{effector}.txt G /flush5/obr17q/offtargets_{gene}_{effector}.txt')


def save_offtarget_count(guides, gene, effector):
    if effector == 'cas9':
        guides = [f'{g[0][:-3]}NNN' for g in guides]
    elif effector == 'cas12a':
        guides = [f'NNNN{g[0][4:]}' for g in guides]

    counts = {g: [0, 0, 0, 0, 0, 0] for g in guides}
    pathlist = Path('/flush5/obr17q').rglob(f'offtargets_{gene}_{effector}_*')
    for path in pathlist:
        with open(path) as f:
            for line in f:
                vals = line.rstrip().split()
                counts[vals[0]][int(vals[8])] += 1

    with open(f'/scratch1/obr17q/counts_{gene}_{effector}.txt', 'w') as f:
        for k, v in counts.items():
            if v[0] > 0:
                v[0] -= 1
            f.write(k + ' ' + ' '.join(map(str, v)) + '\n')


def process_summary(guides, gene, effector):
    with open(f'/scratch1/obr17q/counts_{gene}_{effector}.txt') as f:
        split_lines = [line.rstrip().split() for line in f]

    if effector == 'cas9':
        guide_sequences = [g[0][:-3] for g in guides]
        counts = {s[0][:-3]: [int(o) for o in s[1:]] for s in split_lines}
    elif effector == 'cas12a':
        guide_sequences = [g[0][4:] for g in guides]
        counts = {s[0][4:]: [int(o) for o in s[1:]] for s in split_lines}
    counts = {c: counts[c] for c in counts if c in guide_sequences}
    return(counts)


def run_gtscan_targets(guides, effector):
    targets = [[g[4], g[1], g[2], g[0]] for g in guides]
    data = json.dumps({**gtscan_params, 'targets': targets, 'rule': cas_params[effector]['gtscan_rule'], 'filter': cas_params[effector]['gtscan_filter']})
    submit_res = requests.post(url = gt_url + '/api/submit/', data = data)
    job_id = submit_res.json()['JobID']
    print(f'Submitted job {job_id}. Awaiting completion...')
    job_id = submit_res.json()['JobID']
    finished = 0
    while finished == 0:
        status_res = requests.get(url = f'{gt_url}/api/{job_id}/status/')
        status = status_res.json()['data']
        if (status['statusCode'] == 8):
            finished = 1
        else:
            time.sleep(2)
    results_res = requests.get(url = f'{gt_url}/api/{job_id}/targets/')
    counts = {cleanhtml(r[2])[3:-3]: [int(v) for v in r[3:]] for r in results_res.json()['data']}
    summary = {c: [j if j >= sum(counts[c][:i]) - 100 else 200 for i, j in enumerate(counts[c])] for c in counts}
    print(f'Job completed. See {gt_url}/gt-scan/{job_id} for full results.')
    return summary

def print_results(offtargets):
    for count in offtargets.values():
        print(count)
    counts = [sum(x) for x in zip(*offtargets.values())]
    average = [sum(x) / len(x) for x in zip(*offtargets.values())]
    print(average)


def compare_results(gtscan, casoffinder, effector):
    differences = {}
    for i in gtscan:
        key = i[:-3] if effector == 'cas9' else i[4:]
        ot1 = gtscan[i]
        ot2 = casoffinder[key][:4]
        d = [abs(i[0] - i[1]) for i in zip(ot1, ot2)]
        #d = [abs(i[0] - i[1]) / max(i[0], i[1]) if abs(i[0] - i[1]) else 0  for i in zip(ot1, ot2)]
        differences[key] = d
    return(differences)
    
def exon_gc_content(fasta, gene):
    with open('genes.json') as f:
        coords = json.load(f)['genes']
    gene_info = next(c for c in coords if c['name'] == gene)
    exons = [[min(e['start'], e['end']), max(e['start'], e['end']) - 1] for e in gene_info['exons']]
    print('Gene GC:', gc_content(fasta['sequence']))
    exon_content = []
    for e in exons:
        c = [location - fasta['position'] for location in e]
        seq = fasta['sequence'][c[0]: c[1]]
        exon_content.append(gc_content(seq))
    print('Exon GC:', sum(exon_content) / len(exon_content) )


#genes = ['TNF']
genes = ['TP53', 'TNF', 'EGFR', 'VEGFA', 'APOE', 'IL6', 'TGFB1', 'MTHFR', 'ESR1', 'AKT1']
#genes = ['EGFR', 'VEGFA', 'APOE', 'IL6', 'TGFB1', 'MTHFR', 'ESR1', 'AKT1']

#genes = ['TP53', 'TNF']
#genes = ['TNF']
#genes = ['CAS12A']
effectors = ['cas9']

differences = {}
all_offtargets = {}
all_gtscan_ots = {}
all_casoffinder_ots = {}
for gene in genes:
    fasta = parse_fasta(gene)
    for effector in effectors:
        ## Find guides and filter for just those in exons
        guides = find_guides(fasta, effector)
        #print(guides)
        guides = filter_exonic_guides(guides, gene)

        #print('Total guides:', len(guides))
        #print('Unique guides:', len(set(g[0] for g in guides)))
        #print('Unique spacers:', len(set(g[3] for g in guides)))
        #exon_gc_content(fasta, gene)

        ## One line to get summary of OTs from GT-Scan
        gtscan_ots = run_gtscan_targets(guides, effector)



        ## Multi steps to get summary of OTs from Cas-offinder
#        save_casoffinder_input(guides, effector) #1 - Run this, then run Casoffinder using the output script
#        save_offtarget_count(guides, gene, effector) #2 - After running Casoffinder, run this to create a nice small summary file
#        print(effector)
        casoffinder_ots = process_summary(guides, gene, effector) #3 - Summarise the summary file



        #print_results(casoffinder_ots)
        

        all_gtscan_ots.update({c: gtscan_ots[c] for c in gtscan_ots})
        all_casoffinder_ots.update({c: casoffinder_ots[c][:4] for c in casoffinder_ots})

        differences.update(compare_results(gtscan_ots, casoffinder_ots, effector))

print(differences)

just_bads = {d: differences[d] for d in differences if differences[d].count(0) < 4}



#difffs = [all_offtargets[d] for d in all_offtargets]



#for i in all_gtscan_ots:
#    ot = all_casoffinder_ots[i[:20]][:4]
#    print(f'{ot[0]} {ot[1]} {ot[2]} {ot[3]}')

#for i in all_gtscan_ots:
#    ot = all_gtscan_ots[i]
#    print(f'{ot[0]} {ot[1]} {ot[2]} {ot[3]}')






for i in [0,1,2,3]:
    l = [just_bads[b][i] for b in just_bads if just_bads[b][i]]
    print(f'For {i}:')
    print('Count:  ', len(l))
    print('Sum:    ', sum(l))
    print('Mean:   ', statistics.mean(l) if l else 'NA')
    print('Median: ', statistics.median(l) if l else 'NA')
#    print(statistics.mean([d[i] for d in difffs]))



#for i in just_bads:
#    print(i, just_bads[i])

print(len(just_bads), len(differences))
#print([statistics.median(i) for i in zip(*just_bads)])
#incorrects = [sum(i) for i in just_bads]
#print(len([i for i in incorrects if i == 1]))
#print(len(incorrects))









