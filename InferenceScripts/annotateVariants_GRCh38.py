import sys, getopt, os

# number of rows to pull out of the very large per-position annotation files at a time.
# these files are streamed rather than loaded whole, which keeps peak memory to a few
# hundred MB instead of several GB.
CHUNKSIZE=2000000

# stand-in chromosome number for the alt/random/decoy contigs that the GRCh38 annotation
# files carry. the original code dropped those rows outright; giving them a number no
# sample can ever carry means they simply never match a lookup key.
ALTCONTIG=1000

def chromToInt ( value ):
	if '_' in value:
		return ALTCONTIG
	if value=='X':
		return 23
	if value=='Y':
		return 24
	if value=='MT':
		return 25
	return int(value)


def encodeKeys ( chroms, positions ):
	# pack a chromosome number and a position into a single int64 so that lookups can be
	# done with one sorted array instead of a two column merge
	import numpy as np
	return (chroms.astype(np.int64) << 32) | positions.astype(np.int64)


def buildPositionTable ( fileName, columnNames, chromColumn, posColumn, valueColumn, neededKeys, hasHeader ):
	"""Scan one of the large per-position annotation files and return (keys,values) for the
	positions in neededKeys, keeping the highest value seen at each position. This mirrors
	the sort_values/drop_duplicates + .max() behaviour of the original code, but streams the
	file in chunks and keeps only the handful of rows the sample actually needs, so no
	multi-hundred-million row sort is required."""
	import pandas
	import numpy as np
	keyParts=[]
	valueParts=[]
	if len(neededKeys)==0:
		return np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.float64)
	reader=pandas.read_csv(fileName,sep='\t',header=0 if hasHeader else None,names=columnNames,
			engine='c',dtype={chromColumn:'category',posColumn:np.int32,valueColumn:np.float32},
			chunksize=CHUNKSIZE)
	for chunk in reader:
		chromCodes=chunk[chromColumn].cat.codes.to_numpy()
		chromLookup=np.array([chromToInt(category) for category in chunk[chromColumn].cat.categories],dtype=np.int64)
		keys=(chromLookup[chromCodes] << 32) | chunk[posColumn].to_numpy(dtype=np.int64)
		insertionPoints=np.searchsorted(neededKeys,keys)
		np.clip(insertionPoints,0,len(neededKeys)-1,out=insertionPoints)
		matched=neededKeys[insertionPoints]==keys
		if matched.any():
			keyParts.append(keys[matched])
			valueParts.append(chunk[valueColumn].to_numpy()[matched])
	if len(keyParts)==0:
		return np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.float64)
	keys=np.concatenate(keyParts)
	values=np.concatenate(valueParts)
	# highest value at each position, ignoring NaNs
	best=pandas.Series(values).groupby(keys).max()
	return best.index.to_numpy(dtype=np.int64),best.to_numpy(dtype=np.float64)


def valuesForKeys ( tableKeys, tableValues, queryKeys ):
	# look up one value per query key, NaN where the key is absent
	import numpy as np
	result=np.full(len(queryKeys),np.nan,dtype=np.float64)
	if len(tableKeys)>0 and len(queryKeys)>0:
		insertionPoints=np.searchsorted(tableKeys,queryKeys)
		np.clip(insertionPoints,0,len(tableKeys)-1,out=insertionPoints)
		matched=tableKeys[insertionPoints]==queryKeys
		result[matched]=tableValues[insertionPoints[matched]]
	return result


def maxOverWindows ( tableKeys, tableValues, windowRows, windowKeys, numWindows ):
	# highest value found anywhere inside each variant's window, NaN if the window is empty
	# of annotated positions
	import pandas
	import numpy as np
	result=np.full(numWindows,np.nan,dtype=np.float64)
	if len(windowKeys)>0:
		values=valuesForKeys(tableKeys,tableValues,windowKeys)
		best=pandas.Series(values).groupby(windowRows).max()
		result[best.index.to_numpy()]=best.to_numpy()
	return result


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
	sample['hg38_chr']=sample['hg38_chr'].astype(int)

	# merge on the allele frequency data. only the rows of the gnomad file that sit on one of
	# the sample's positions can contribute to the merge, so the file is streamed and filtered
	# down before the merge instead of being loaded in full.
	print('Merging on allele frequency data')
	sampleAFKeys=np.unique(encodeKeys(sample['hg38_chr'].to_numpy(),sample['hg38_pos(1-based)'].to_numpy()))
	gnomadAFParts=[]
	reader=pandas.read_csv('gnomad211_GRCh38_exomes_AFs.txt',sep='\t',engine='c',
			dtype={'hg38_chr':'category','hg38_pos(1-based)':np.int32,'ref':str,'alt':str,'AF':np.float32,
					'nhomalt':np.int32,'controls_AF':np.float32,'controls_nhomalt':np.int32},chunksize=CHUNKSIZE)
	for chunk in reader:
		chromCodes=chunk['hg38_chr'].cat.codes.to_numpy()
		chromLookup=np.array([chromToInt(category) for category in chunk['hg38_chr'].cat.categories],dtype=np.int64)
		chroms=chromLookup[chromCodes]
		keys=(chroms << 32) | chunk['hg38_pos(1-based)'].to_numpy(dtype=np.int64)
		insertionPoints=np.searchsorted(sampleAFKeys,keys)
		np.clip(insertionPoints,0,len(sampleAFKeys)-1,out=insertionPoints)
		matched=sampleAFKeys[insertionPoints]==keys
		if matched.any():
			part=chunk.loc[matched,:].copy()
			part['hg38_chr']=chroms[matched]
			gnomadAFParts.append(part)
	if len(gnomadAFParts)>0:
		gnomadAF=pandas.concat(gnomadAFParts,axis=0,ignore_index=True)
	else:
		gnomadAF=pandas.DataFrame({'hg38_chr':np.zeros(0,dtype=np.int64),'hg38_pos(1-based)':np.zeros(0,dtype=np.int32),
				'ref':pandas.Series(dtype=str),'alt':pandas.Series(dtype=str),'AF':np.zeros(0,dtype=np.float32),
				'nhomalt':np.zeros(0,dtype=np.int32),'controls_AF':np.zeros(0,dtype=np.float32),
				'controls_nhomalt':np.zeros(0,dtype=np.int32)})
	sample=sample.merge(gnomadAF,how='left',on=['hg38_chr','hg38_pos(1-based)','ref','alt'])

	# merge on the constraint data (try transcript ID merge first)
	print('Merging on constraint data')
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

	# SNVs take the value sitting on their own position, while indels take the highest value
	# found in the window that starts one base after the variant and is as long as the
	# reference allele
	sampleSNVs=sample2.loc[sample2['varType'].isin(['nonsynonymous SNV','synonymous SNV','stopgain','stoploss']),['hg38_chr','hg38_pos(1-based)']]
	sampleIndels=sample2.loc[sample2['varType'].isin(['frameshift insertion','frameshift deletion','frameshift substitution',
													'nonframeshift insertion','nonframeshift deletion','nonframeshift substitution']),['hg38_chr','hg38_pos(1-based)','ref']].copy()
	sampleIndels['length']=sampleIndels['ref'].str.len()

	snvKeys=encodeKeys(sampleSNVs['hg38_chr'].to_numpy(),sampleSNVs['hg38_pos(1-based)'].to_numpy())
	# flatten every indel window into one long list of (indel row, position) pairs
	indelLengths=sampleIndels['length'].to_numpy(dtype=np.int64)
	windowRows=np.repeat(np.arange(len(sampleIndels)),indelLengths)
	windowOffsets=np.arange(int(indelLengths.sum()),dtype=np.int64)-np.repeat(np.cumsum(indelLengths)-indelLengths,indelLengths)
	windowPositions=np.repeat(sampleIndels['hg38_pos(1-based)'].to_numpy(dtype=np.int64)+1,indelLengths)+windowOffsets
	windowKeys=(np.repeat(sampleIndels['hg38_chr'].to_numpy(dtype=np.int64),indelLengths) << 32) | windowPositions
	neededKeys=np.unique(np.concatenate([snvKeys,windowKeys]))

	# merge on the CCR data
	print('Merging on CCR data')
	sample2['CCR']=np.nan
	tableKeys,tableValues=buildPositionTable('ccrs_GRCh38.enumerated.txt',['chrom','pos','ccr_pct'],'chrom','pos','ccr_pct',neededKeys,True)
	sample2.loc[sampleSNVs.index,'CCR']=valuesForKeys(tableKeys,tableValues,snvKeys)
	sample2.loc[sampleIndels.index,'CCR']=maxOverWindows(tableKeys,tableValues,windowRows,windowKeys,len(sampleIndels))

	# merge on the pext data
	print('Merging on pext data')
	sample2['pext']=np.nan
	tableKeys,tableValues=buildPositionTable('gnomAD_pext_values_GRCh38.txt',['chr','pos','pext'],'chr','pos','pext',neededKeys,True)
	sample2.loc[sampleSNVs.index,'pext']=valuesForKeys(tableKeys,tableValues,snvKeys)
	sample2.loc[sampleIndels.index,'pext']=maxOverWindows(tableKeys,tableValues,windowRows,windowKeys,len(sampleIndels))

	# merge on the GERP data
	print('Merging on GERP data')
	sample2['gerp']=np.nan
	tableKeys,tableValues=buildPositionTable('gerpOnExons_GRCh38.txt',['chr','pos','gerp'],'chr','pos','gerp',neededKeys,False)
	sample2.loc[sampleSNVs.index,'gerp']=valuesForKeys(tableKeys,tableValues,snvKeys)
	sample2.loc[sampleIndels.index,'gerp']=maxOverWindows(tableKeys,tableValues,windowRows,windowKeys,len(sampleIndels))

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

