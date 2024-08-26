import sys, getopt, os

def main ( argv ):
	base = ''
	try:
		opts, args = getopt.getopt(argv,"h",["inputBase=","help"])
	except getopt.GetoptError:
		print('groomAnnovarOutput.py --inputBase=<baseName>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('--inputBase'):
			base=arg
		elif opt in ('-h','--help'):
			print('groomAnnovarOutput.py --inputBase=<baseName>')
			sys.exit()
		else:
			print('groomAnnovarOutput.py --inputBase=<baseName>')
			sys.exit()

	import pandas
	import numpy as np
	from Bio import SeqIO
	approvedTranscripts=pandas.read_csv('gencodeBasicFullLengthTranscriptsConversionTable.txt',sep='\t',low_memory=False)

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
		transcriptID=record.id.split('|')[1]
		if len(approvedTranscripts.loc[approvedTranscripts['transcriptID']==transcriptID,:])>0:
			sequences[transcriptID]=record.seq

	# load in the sample data (expect indels to be in 'long form' already)
	sample=pandas.read_csv(base + ".avinput.exonic_variant_function",sep='\t',low_memory=False,header=None,
						names=['line','varType','location','hg19_chr','hg19_pos(1-based)','end','ref','alt','genotype','qual','depth'])
	# add new columns with placeholders to be filled in
	sample['WildtypeSeq']=""
	sample['AltSeq']=""
	sample['ChangePos']=-1
	sample['TranscriptID']=""
	sample['TranscriptIDShort']=sample['location'].str.split(':',expand=True)[1].str[:15]
	sample['geneName']=sample['location'].str.split(':',expand=True)[0]
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
			if sample.loc[i,'location'].split(',')[j].split(':')[1] in approvedTranscripts['transcriptID'].values:  
				transcripts.append(sample.loc[i,'location'].split(',')[j].split(':')[1][:15])
				transcriptLengths.append(len(sequences[sample.loc[i,'location'].split(',')[j].split(':')[1]]))
				
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

	for record in SeqIO.parse(base + ".coding_changes.txt", "fasta"):
		lineNum=record.id
		# only use the transcript that we selected above 
		if sample.loc[sample['line']==lineNum,'TranscriptID'].values==record.description.split(' ')[1]:
			if 'WILDTYPE' in record.description:
				if record.seq.__str__()[:-1] == sequences[record.description.split(' ')[1]]:
					sample.loc[sample['line']==lineNum,'WildtypeSeq']=record.seq.__str__()
					sample.loc[sample['line']==lineNum,'TranscriptID']=record.description.split(' ')[1]
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

