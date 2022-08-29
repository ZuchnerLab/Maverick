import sys, getopt, os

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
	from Bio import SeqIO
	approvedTranscripts=pandas.read_csv('gencodeBasicFullLengthTranscriptsConversionTable_GRCh38.txt',sep='\t',low_memory=False)

	canonical=pandas.read_csv('gnomad211_constraint_canonical_simple.txt',sep='\t',low_memory=False)
	# remove the gnomad canonical transcripts that are not approvedTranscripts
	canonical=canonical.loc[canonical['transcript'].isin(approvedTranscripts['transcriptIDShort'].values),:].reset_index(drop=True)

	GTEx=pandas.read_csv('GTEx.V7.tx_medians.021820.tsv',sep='\t',low_memory=False)
	# remove the non-approvedTranscripts from the expression data
	GTEx=GTEx.loc[GTEx['transcript_id'].isin(approvedTranscripts['transcriptIDShort'].values),:].reset_index(drop=True)
	# add a overall expression column
	GTEx['overallAvg']=GTEx.iloc[:,2:55].mean()

	sequences={}
	for record in SeqIO.parse("gencode.v33lift37.pc_translations.fa","fasta"):
		transcriptID=record.id.split('|')[1][:15]
		if len(approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,:])>0:
			sequences[transcriptID]=record.seq

	sample=pandas.read_csv(base + ".avinput.exonic_variant_function",sep='\t',low_memory=False,header=None,
						names=['line','varType','location','hg38_chr','hg38_pos(1-based)','end','ref','alt','genotype','qual','depth'])
	# convert the position, ref, and alt alleles to long form
	longForm=pandas.read_csv(base + "_locations.txt",sep='\t',low_memory=False,header=None,names=['chrom','pos_long','ref_long','alt_long'])
	sample['lineNum']=sample.loc[:,'line'].str[4:].astype(int)-1
	sample=sample.merge(longForm,how='inner',left_on='lineNum',right_on=longForm.index)
	sample=sample.loc[:,['line','varType','location','hg38_chr','pos_long','end','ref_long','alt_long','genotype','qual','depth']].rename(columns={'pos_long':'hg38_pos(1-based)','ref_long':'ref','alt_long':'alt'}).reset_index(drop=True)
	# add new columns with placeholders to be filled in
	sample['WildtypeSeq']=""
	sample['AltSeq']=""
	sample['ChangePos']=-1
	sample['TranscriptID']=""
	sample['TranscriptIDShort']=""
	sample['geneName']=""
	sample['geneID']=""
	sample['geneIDShort']=""
	

	for i in range(len(sample)):
		if i % 1000 == 0:
			print(str(i) + ' rows completed')
		numTranscripts=len(sample.loc[i,'location'].split(','))
		numCanonical=0
		canonicals=[]
		transcripts=[]
		transcriptLengths=[]
		canonicalTranscript=""
		correctedGeneName=""
		for j in range(numTranscripts-1):
			if sample.loc[i,'location'].split(',')[j].split(':')[1][:15] in canonical['transcript'].values:
				numCanonical=numCanonical+1
				canonicals.append(sample.loc[i,'location'].split(',')[j].split(':')[1][:15])
			if sample.loc[i,'location'].split(',')[j].split(':')[1][:15] in approvedTranscripts['transcriptIDShort'].values:  
				transcripts.append(sample.loc[i,'location'].split(',')[j].split(':')[1][:15])
				transcriptLengths.append(len(sequences[sample.loc[i,'location'].split(',')[j].split(':')[1][:15]]))
				
		if len(transcripts)>0:
			if numCanonical==1:
				transcriptID=canonicals[0]
				sample.loc[i,'TranscriptIDShort']=transcriptID
				sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
				sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
				sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
				sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]
			elif numCanonical==0:
				if len(transcripts)==1:
					transcriptID=transcripts[0]
					sample.loc[i,'TranscriptIDShort']=transcriptID
					sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
					sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
					sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
					sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]
				else:
					if len(GTEx.loc[GTEx['transcript_id'].isin(transcripts),:])>0:
						# pick the transcript with the highest expression
						transcriptID=GTEx.loc[GTEx['transcript_id'].isin(transcripts),:].sort_values(by=['overallAvg'],ascending=False).reset_index(drop=True).iloc[0,0]
						sample.loc[i,'TranscriptIDShort']=transcriptID
						sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
						sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
						sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
						sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]
					else:
						# if none of the transcripts have measured expression and none of them are canonical, then pick the one with the longest amino acid sequence
						# if multiple tie for longest, this picks the one we saw first
						j=transcriptLengths.index(max(transcriptLengths))
						transcriptID=transcripts[j]
						sample.loc[i,'TranscriptIDShort']=transcriptID
						sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
						sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
						sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
						sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]
			elif numCanonical>1:
				if len(GTEx.loc[GTEx['transcript_id'].isin(canonicals),:])>0:
					# pick the canonical transcript with the highest expression
					transcriptID=GTEx.loc[GTEx['transcript_id'].isin(canonicals),:].sort_values(by=['overallAvg'],ascending=False).reset_index(drop=True).iloc[0,0]
					sample.loc[i,'TranscriptIDShort']=transcriptID
					sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
					sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
					sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
					sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]
				else:
					# if none of the canonical transcripts have measured expression, then pick the one with the longest amino acid sequence
					# if multiple tie for longest, this picks the one we saw first
					j=transcriptLengths.index(max(transcriptLengths))
					transcriptID=transcripts[j]
					sample.loc[i,'TranscriptIDShort']=transcriptID
					sample.loc[i,'TranscriptID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'transcriptID'].values[0]
					sample.loc[i,'geneName']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneName'].values[0]
					sample.loc[i,'geneID']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneID'].values[0]
					sample.loc[i,'geneIDShort']=approvedTranscripts.loc[approvedTranscripts['transcriptIDShort']==transcriptID,'geneIDShort'].values[0]

	# drop the entries without transcript ID matches
	sample=sample.loc[(~(sample['TranscriptIDShort']=="")),:].reset_index(drop=True)

	for record in SeqIO.parse(base + ".coding_changes.txt", "fasta"):
		lineNum=record.id
		# only use the transcript that we selected above 
		if ((len(sample.loc[sample['line']==lineNum,:])>0) and (sample.loc[sample['line']==lineNum,'TranscriptIDShort'].values[0]==record.description.split(' ')[1][:15])):
			if 'WILDTYPE' in record.description:
				if record.seq.__str__()[:-1] == sequences[record.description.split(' ')[1][:15]]:
					sample.loc[sample['line']==lineNum,'WildtypeSeq']=record.seq.__str__()
			else:
				sample.loc[sample['line']==lineNum,'AltSeq']=record.seq.__str__()
				if 'startloss' in record.description:
					sample.loc[sample['line']==lineNum,'ChangePos']=1
				elif 'silent' in record.description:
					sample.loc[sample['line']==lineNum,'ChangePos']=-1
				else:
					sample.loc[sample['line']==lineNum,'ChangePos']=record.description.split(' ')[7].split('-')[0]
	sample2=sample.loc[~((sample['WildtypeSeq']=="") | (sample['AltSeq']=="") | (sample['ChangePos']==-1)),:]
	sample2.to_csv(base + '.groomed.txt',sep='\t',index=False)
	return


if __name__ == "__main__":
	main(sys.argv[1:])

