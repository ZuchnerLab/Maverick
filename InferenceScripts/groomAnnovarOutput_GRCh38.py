import sys, getopt, os

def readFasta ( fileName ):
	# minimal FASTA reader yielding (id, description, sequence) tuples.
	# equivalent to Bio.SeqIO.parse(...,'fasta') for the fields used here, but
	# avoids building a SeqRecord/Seq object for every one of the ~100k records
	with open(fileName) as handle:
		description=None
		chunks=[]
		for line in handle:
			if line.startswith('>'):
				if description is not None:
					yield description.split(' ',1)[0], description, ''.join(chunks)
				description=line[1:].rstrip()
				chunks=[]
			elif description is not None:
				chunks.append(line.strip())
		if description is not None:
			yield description.split(' ',1)[0], description, ''.join(chunks)
	return


def main ( argv ):
	base = ''
	try:
		opts, args = getopt.getopt(argv,"h",["inputBase=","help"])
	except getopt.GetoptError:
		print('groomAnnovarOutput_GRCh38.py --inputBase=<baseName>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('--inputBase'):
			base=arg
		elif opt in ('-h','--help'):
			print('groomAnnovarOutput_GRCh38.py --inputBase=<baseName>')
			sys.exit()
		else:
			print('groomAnnovarOutput_GRCh38.py --inputBase=<baseName>')
			sys.exit()

	import pandas
	import numpy as np
	approvedTranscripts=pandas.read_csv('gencodeBasicFullLengthTranscriptsConversionTable_GRCh38.txt',sep='\t',low_memory=False)
	# the hg38 Gencode annovar table names transcripts without the trailing _N suffix, so
	# everything here is matched on the 15 character short transcript ID.
	# map each short transcript ID to the first row of the conversion table that carries it.
	# 'first' matches the .values[0] indexing that the original implementation used, which
	# matters for the handful of short IDs that appear under more than one version.
	approvedByShortID={}
	for transcriptID,transcriptIDShort,geneName,geneID,geneIDShort in zip(approvedTranscripts['transcriptID'].values,
			approvedTranscripts['transcriptIDShort'].values,approvedTranscripts['geneName'].values,
			approvedTranscripts['geneID'].values,approvedTranscripts['geneIDShort'].values):
		if transcriptIDShort not in approvedByShortID:
			approvedByShortID[transcriptIDShort]=(transcriptID,geneName,geneID,geneIDShort)

	canonical=pandas.read_csv('gnomad211_constraint_canonical_simple.txt',sep='\t',low_memory=False)
	# remove the gnomad canonical transcripts that are not approvedTranscripts
	canonicalTranscripts=set(canonical.loc[canonical['transcript'].isin(approvedByShortID.keys()),'transcript'].values)

	# The GTEx table is only consulted to break ties between candidate transcripts. The
	# original code sorted the candidates by an 'overallAvg' column built with
	# GTEx.iloc[:,2:55].mean(), which produces a Series indexed by tissue name; assigning
	# that back to a column indexed 0..n-1 leaves every value NaN, so the sort is a no-op
	# and the winner is simply the candidate that appears first in the (approved-transcript
	# filtered) GTEx table. Only that row ordering is reproduced here, so we can read the
	# single column we need instead of the whole 120MB expression matrix.
	GTExRank={}
	filteredRank=0
	for transcript_id in pandas.read_csv('GTEx.V7.tx_medians.021820.tsv',sep='\t',usecols=['transcript_id'],low_memory=False)['transcript_id'].values:
		if transcript_id in approvedByShortID:
			if transcript_id not in GTExRank:
				GTExRank[transcript_id]=filteredRank
			filteredRank=filteredRank+1

	# keyed on the short transcript ID, so where the fasta holds more than one version of a
	# transcript the last record read wins, exactly as the original dict assignment did
	sequences={}
	for recordID,description,sequence in readFasta("gencode.v33lift37.pc_translations.fa"):
		transcriptIDShort=recordID.split('|')[1][:15]
		if transcriptIDShort in approvedByShortID:
			sequences[transcriptIDShort]=sequence

	# load in the sample data (expect indels to be in 'long form' already)
	sample=pandas.read_csv(base + ".avinput.exonic_variant_function",sep='\t',low_memory=False,header=None,
						names=['line','varType','location','hg38_chr','hg38_pos(1-based)','end','ref','alt','genotype','qual','depth'])
	# add new columns with placeholders to be filled in
	sample['WildtypeSeq']=""
	sample['AltSeq']=""
	sample['ChangePos']=-1
	sample['TranscriptID']=""
	sample['TranscriptIDShort']=""
	sample['geneName']=""
	sample['geneID']=""
	sample['geneIDShort']=""

	# work on plain object arrays rather than assigning into the DataFrame cell by cell,
	# then write each finished array back as a whole column at the end
	numRows=len(sample)
	transcriptIDShortCol=np.full(numRows,"",dtype=object)
	transcriptIDCol=np.full(numRows,"",dtype=object)
	geneNameCol=np.full(numRows,"",dtype=object)
	geneIDCol=np.full(numRows,"",dtype=object)
	geneIDShortCol=np.full(numRows,"",dtype=object)

	locations=sample['location'].to_numpy(dtype=object)
	for i in range(numRows):
		if i % 10000 == 0:
			print(str(i) + ' rows completed of ' + str(numRows))
		location=locations[i]
		if not isinstance(location,str):
			continue
		canonicals=[]
		transcripts=[]
		transcriptLengths=[]
		# the location field carries a trailing comma, so the last piece is always empty
		for annotation in location.split(',')[:-1]:
			fields=annotation.split(':')
			if len(fields)<2:
				continue
			transcriptIDShort=fields[1][:15]
			if transcriptIDShort in canonicalTranscripts:
				canonicals.append(transcriptIDShort)
			if transcriptIDShort in approvedByShortID:
				transcripts.append(transcriptIDShort)
				transcriptLengths.append(len(sequences[transcriptIDShort]))

		if len(transcripts)>0:
			numCanonical=len(canonicals)
			if numCanonical==1:
				transcriptID=canonicals[0]
			elif numCanonical==0:
				if len(transcripts)==1:
					transcriptID=transcripts[0]
				else:
					# pick the transcript with the highest expression
					transcriptID=pickByExpression(transcripts,GTExRank)
					if transcriptID is None:
						# if none of the transcripts have measured expression and none of them are canonical, then pick the one with the longest amino acid sequence
						# if multiple tie for longest, this picks the one we saw first
						transcriptID=transcripts[transcriptLengths.index(max(transcriptLengths))]
			else:
				# pick the canonical transcript with the highest expression
				transcriptID=pickByExpression(canonicals,GTExRank)
				if transcriptID is None:
					# if none of the canonical transcripts have measured expression, then pick the one with the longest amino acid sequence
					# if multiple tie for longest, this picks the one we saw first
					transcriptID=transcripts[transcriptLengths.index(max(transcriptLengths))]

			transcriptInfo=approvedByShortID.get(transcriptID)
			if transcriptInfo is not None:
				transcriptIDShortCol[i]=transcriptID
				transcriptIDCol[i]=transcriptInfo[0]
				geneNameCol[i]=transcriptInfo[1]
				geneIDCol[i]=transcriptInfo[2]
				geneIDShortCol[i]=transcriptInfo[3]

	sample['TranscriptID']=transcriptIDCol
	sample['TranscriptIDShort']=transcriptIDShortCol
	sample['geneName']=geneNameCol
	sample['geneID']=geneIDCol
	sample['geneIDShort']=geneIDShortCol

	# drop the entries without transcript ID matches
	sample=sample.loc[(~(sample['TranscriptIDShort']=="")),:].reset_index(drop=True)

	# map each annovar line number to the row(s) of the sample that carry it so that the
	# coding_changes records can be matched without rescanning the whole table each time
	numRows=len(sample)
	transcriptIDShortCol=sample['TranscriptIDShort'].to_numpy(dtype=object)
	rowsByLine={}
	for rowIndex,lineNum in enumerate(sample['line'].to_numpy(dtype=object)):
		rowsByLine.setdefault(lineNum,[]).append(rowIndex)

	wildtypeSeqCol=np.full(numRows,"",dtype=object)
	altSeqCol=np.full(numRows,"",dtype=object)
	changePosCol=np.full(numRows,-1,dtype=object)

	for recordID,description,sequence in readFasta(base + ".coding_changes.txt"):
		rows=rowsByLine.get(recordID)
		if rows is None:
			continue
		descriptionFields=description.split(' ')
		if len(descriptionFields)<2:
			continue
		recordTranscriptIDShort=descriptionFields[1][:15]
		# only use the transcript that we selected above
		if transcriptIDShortCol[rows[0]]!=recordTranscriptIDShort:
			continue
		for i in rows:
			if 'WILDTYPE' in description:
				if sequence[:-1] == sequences.get(recordTranscriptIDShort):
					wildtypeSeqCol[i]=sequence
			else:
				altSeqCol[i]=sequence
				if 'startloss' in description:
					changePosCol[i]=1
				elif 'silent' in description:
					changePosCol[i]=-1
				elif len(descriptionFields)>7:
					changePosCol[i]=descriptionFields[7].split('-')[0]

	sample['WildtypeSeq']=wildtypeSeqCol
	sample['AltSeq']=altSeqCol
	sample['ChangePos']=changePosCol

	sample2=sample.loc[~((sample['WildtypeSeq']=="") | (sample['AltSeq']=="") | (sample['ChangePos']==-1)),:]
	sample2.to_csv(base + '.groomed.txt',sep='\t',index=False)
	return


def pickByExpression ( candidates, GTExRank ):
	# return the candidate that comes first in the GTEx expression table, or None if
	# none of the candidates appear in it
	best=None
	bestRank=None
	for candidate in candidates:
		rank=GTExRank.get(candidate)
		if rank is not None and (bestRank is None or rank<bestRank):
			best=candidate
			bestRank=rank
	return best


if __name__ == "__main__":
	main(sys.argv[1:])

