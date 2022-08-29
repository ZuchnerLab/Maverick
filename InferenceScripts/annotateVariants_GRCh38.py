import sys, getopt, os

def main ( argv ):
	base = ''
	try:
		opts, args = getopt.getopt(argv,"h",["inputBase=","help"])
	except getopt.GetoptError:
		print('annotateVariants_GRCh38.py --inputBase <baseName>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('--inputBase'):
			base=arg
		elif opt in ('-h','--help'):
			print('annotateVariants_GRCh38.py --inputBase <baseName>')
			sys.exit()
		else:
			print('annotateVariants_GRCh38.py --inputBase <baseName>')
			sys.exit()

	import pandas
	import numpy as np
	sample=pandas.read_csv(base + '.groomed.txt',sep='\t',low_memory=False,dtype={'hg38_chr':str})
	sample.loc[sample['hg38_chr']=='X','hg38_chr']=23
	sample.loc[sample['hg38_chr']=='Y','hg38_chr']=24
	sample.loc[sample['hg38_chr']=='MT','hg38_chr']=25
	sample.loc[(~(sample['hg38_chr'].str.contains('_'))),:].reset_index(drop=True)
	sample['hg38_chr']=sample['hg38_chr'].astype(int)

	# merge on the allele frequency data
	gnomadAF=pandas.read_csv('gnomad211_GRCh38_exomes_AFs.txt',sep='\t',low_memory=False,dtype={'hg38_chr':str,'hg38_pos(1-based)':np.int32,'ref':str,'alt':str,'AF':np.float32,'nhomalt':np.int32,'controls_AF':np.float32,'controls_nhomalt':np.int32})
	gnomadAF=gnomadAF.loc[(~(gnomadAF['hg38_chr'].str.contains('_'))),:].reset_index(drop=True)
	gnomadAF.loc[gnomadAF['hg38_chr']=='X','hg38_chr']=23
	gnomadAF.loc[gnomadAF['hg38_chr']=='Y','hg38_chr']=24
	gnomadAF.loc[gnomadAF['hg38_chr']=='MT','hg38_chr']=25
	gnomadAF['hg38_chr']=gnomadAF['hg38_chr'].astype(int)
	sample=sample.merge(gnomadAF,how='left',on=['hg38_chr','hg38_pos(1-based)','ref','alt'])

	# merge on the constraint data (try transcript ID merge first)
	constraint=pandas.read_csv('gnomad211_constraint_simple.txt',sep='\t',low_memory=False)
	sampleTranscript=sample.merge(constraint,how='inner',left_on=['TranscriptIDShort'],right_on=['transcript'])
	notMatched=sample.loc[~(sample['TranscriptIDShort'].isin(sampleTranscript['TranscriptIDShort'])),:]
	constraintGeneLevel=pandas.read_csv('gnomad211_constraint_simple_geneLevel.txt',sep='\t',low_memory=False)
	sampleGeneID=notMatched.merge(constraintGeneLevel,how='inner',left_on=['geneIDShort'],right_on=['gene_id'])
	notMatched2=notMatched.loc[~(notMatched['geneIDShort'].isin(sampleGeneID['geneIDShort'])),:]
	sampleGeneName=notMatched2.merge(constraintGeneLevel,how='left',left_on=['geneName'],right_on=['gene'])
	# stack them all back together
	sample2=pandas.concat([sampleTranscript,sampleGeneID,sampleGeneName],axis=0,ignore_index=True)
	sample2.loc[sample2['hg38_chr']=='X','hg38_chr']=23
	sample2.loc[sample2['hg38_chr']=='Y','hg38_chr']=24
	sample2.loc[sample2['hg38_chr']=='MT','hg38_chr']=25
	sample2['hg38_chr']=sample2['hg38_chr'].astype(int)

	# merge on the CCR data
	sample2['CCR']=np.nan
	CCR=pandas.read_csv('ccrs_GRCh38.enumerated.txt',sep='\t',low_memory=False,dtype={'chrom':str,'pos':np.int32,'ccr_pct':np.float32})
	CCR=CCR.loc[(~(CCR['chrom'].str.contains('_'))),:]
	CCR.loc[CCR['chrom']=='X','chrom']=23
	CCR['chrom']=CCR.loc[:,'chrom'].astype(int)
	CCR=CCR.sort_values(by=['chrom','pos','ccr_pct'],ascending=[True,True,False]).drop_duplicates(subset=['chrom','pos'],keep='first').reset_index(drop=True)
	sampleSNVs=sample2.loc[sample2['varType'].isin(['nonsynonymous SNV','synonymous SNV','stopgain','stoploss']),['hg38_chr','hg38_pos(1-based)']]
	sampleIndels=sample2.loc[sample2['varType'].isin(['frameshift insertion','frameshift deletion','frameshift substitution',
													'nonframeshift insertion','nonframeshift deletion','nonframeshift substitution']),['hg38_chr','hg38_pos(1-based)','ref']]
	sampleIndels['length']=sampleIndels['ref'].str.len()
	sampleIndels['CCR']=np.nan
	sampleSNVs2=sampleSNVs.merge(CCR,how='left',left_on=['hg38_chr','hg38_pos(1-based)'],right_on=['chrom','pos']).set_index(sampleSNVs.index)
	for i in range(len(sampleIndels)):
		if i%100==0:
			print(str(i) + ' rows complete of ' + str(len(sampleIndels)))
		startPos=sampleIndels.iloc[i,1]+1
		endPos=startPos+sampleIndels.iloc[i,3]
		sampleIndels.iloc[i,4]=CCR.loc[((CCR['chrom']==sampleIndels.iloc[i,0]) & (CCR['pos'].isin(range(startPos,endPos)))),'ccr_pct'].max()
	sample2.loc[sampleSNVs2.index,'CCR']=sampleSNVs2.loc[:,'ccr_pct'].values
	sample2.loc[sampleIndels.index,'CCR']=sampleIndels.loc[:,'CCR'].values

	# merge on the pext data
	pext=pandas.read_csv('gnomAD_pext_values_GRCh38.txt',sep='\t',low_memory=False,dtype={'chr':str,'pos':np.int32,'pext':np.float32})
	pext=pext.loc[(~(pext['chr'].str.contains('_'))),:]
	pext.loc[pext['chr']=='X','chr']=23
	pext.loc[pext['chr']=='Y','chr']=24
	pext.loc[pext['chr']=='MT','chr']=25
	pext['chr']=pext.loc[:,'chr'].astype(int)
	pext=pext.sort_values(by=['chr','pos','pext'],ascending=[True,True,False]).drop_duplicates(subset=['chr','pos'],keep='first').reset_index(drop=True)
	sample2['pext']=np.nan
	sampleIndels['pext']=np.nan
	sampleSNVs2=sampleSNVs.merge(pext,how='left',left_on=['hg38_chr','hg38_pos(1-based)'],right_on=['chr','pos']).set_index(sampleSNVs.index)
	for i in range(len(sampleIndels)):
		if i%100==0:
			print(str(i) + ' rows complete of ' + str(len(sampleIndels)))
		startPos=sampleIndels.iloc[i,1]+1
		endPos=startPos+sampleIndels.iloc[i,3]
		sampleIndels.iloc[i,5]=pext.loc[((pext['chr']==sampleIndels.iloc[i,0]) & (pext['pos'].isin(range(startPos,endPos)))),'pext'].max()

	sample2.loc[sampleSNVs2.index,'pext']=sampleSNVs2.loc[:,'pext'].values
	sample2.loc[sampleIndels.index,'pext']=sampleIndels.loc[:,'pext'].values

	# merge on the GERP data
	gerp=pandas.read_csv('gerpOnExons_GRCh38.txt',sep='\t',low_memory=False,header=None,names=['chr','pos','gerp'],dtype={'chr':str,'pos':np.int32,'gerp':np.float32})
	gerp=gerp.loc[(~(gerp['chr'].str.contains('_'))),:]
	gerp.loc[gerp['chr']=='X','chr']=23
	gerp.loc[gerp['chr']=='Y','chr']=24
	gerp.loc[gerp['chr']=='MT','chr']=25
	gerp['chr']=gerp['chr'].astype(int)
	gerp=gerp.sort_values(by=['chr','pos','gerp'],ascending=[True,True,False]).drop_duplicates(subset=['chr','pos'],keep='first').reset_index(drop=True)
	sample2['gerp']=np.nan
	sampleIndels['gerp']=np.nan
	sampleSNVs2=sampleSNVs.merge(gerp,how='left',left_on=['hg38_chr','hg38_pos(1-based)'],right_on=['chr','pos']).set_index(sampleSNVs.index)
	for i in range(len(sampleIndels)):
		if i%100==0:
			print(str(i) + ' rows complete of ' + str(len(sampleIndels)))
		startPos=sampleIndels.iloc[i,1]+1
		endPos=startPos+sampleIndels.iloc[i,3]
		sampleIndels.iloc[i,6]=gerp.loc[((gerp['chr']==sampleIndels.iloc[i,0]) & (gerp['pos'].isin(range(startPos,endPos)))),'gerp'].max()

	sample2.loc[sampleSNVs2.index,'gerp']=sampleSNVs2.loc[:,'gerp'].values
	sample2.loc[sampleIndels.index,'gerp']=sampleIndels.loc[:,'gerp'].values

	sample2=sample2.drop_duplicates(subset=['hg38_chr','hg38_pos(1-based)','ref','alt'],keep='first')
	sample2=sample2.drop(columns=['line','location','end','qual','depth','gene','transcript', 'canonical','gene_id'])
	sample2=sample2.sort_values(by=['hg38_chr','hg38_pos(1-based)','ref','alt']).reset_index(drop=True)

	# merge on GDI data
	GDI=pandas.read_csv('GDI.groomed.txt',sep='\t',low_memory=False)
	sample2=sample2.merge(GDI,how='left',on='geneName')
	# merge on RVIS data
	RVIS=pandas.read_csv('RVIS.groomed.txt',sep='\t',low_memory=False)
	sample2=sample2.merge(RVIS,how='left',on='geneName')

	sample2=sample2.sort_values(by=['hg38_chr','hg38_pos(1-based)','ref','alt']).reset_index(drop=True)
	sample2=sample2.drop_duplicates(subset=['hg38_chr','hg38_pos(1-based)','ref','alt'],keep='first').reset_index(drop=True)
	sample2.to_csv(base + '.annotated.txt',sep='\t',index=False)
	return


if __name__ == "__main__":
	main(sys.argv[1:])

