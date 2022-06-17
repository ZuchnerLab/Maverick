import sys, getopt, os

def main ( argv ):
	base = ''
	try:
		opts, args = getopt.getopt(argv,"h",["inputBase=","help"])
	except getopt.GetoptError:
		print('rankScores.py --inputBase <baseName>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('--inputBase'):
			base=arg
		elif opt in ('-h','--help'):
			print('rankScores.py --inputBase <baseName>')
			sys.exit()
		else:
			print('rankScores.py --inputBase <baseName>')
			sys.exit()

	import pandas
	import numpy as np
	import scipy
	from scipy import stats
	sample=pandas.read_csv(base + '.MaverickResults.txt',sep='\t',low_memory=False)
	sample['varID']=sample.loc[:,['hg19_chr','hg19_pos(1-based)','ref','alt']].apply(lambda row: '_'.join(row.values.astype(str)),axis=1)
	sample['TotalScore']=sample.loc[:,'Maverick_DomScore']
	sample.loc[sample['genotype']=='hom','TotalScore']=sample.loc[sample['genotype']=='hom','Maverick_RecScore']
	compHetPairs=pandas.DataFrame(columns=['site1_varID','site2_varID','geneID','geneName','site1_RecScore','site2_RecScore','TotalScore'])
	hets=sample.loc[sample['genotype']=='het',:].reset_index(drop=True)
	hetCallsOnSharedGenes=hets.loc[hets.duplicated(subset='geneID',keep=False),:]
	genesWithMultipleHets=hets.loc[hets.duplicated(subset='geneID',keep='first'),'geneID'].unique()
	for i in range(0,len(genesWithMultipleHets)):
		thisGeneGroup=hetCallsOnSharedGenes.loc[hetCallsOnSharedGenes['geneID']==genesWithMultipleHets[i],:]
		for j in range(0,len(thisGeneGroup)-1):
			for k in range(j+1,len(thisGeneGroup)):
				harmonicMean=scipy.stats.hmean([thisGeneGroup.loc[thisGeneGroup.index[j],'Maverick_RecScore'],thisGeneGroup.loc[thisGeneGroup.index[k],'Maverick_RecScore']])
				compHetPairs=pandas.concat([compHetPairs,pandas.DataFrame({'site1_varID':thisGeneGroup.loc[thisGeneGroup.index[j],'varID'],
					'site2_varID':thisGeneGroup.loc[thisGeneGroup.index[k],'varID'],
					'geneID':thisGeneGroup.loc[thisGeneGroup.index[k],'geneID'],
					'geneName':thisGeneGroup.loc[thisGeneGroup.index[k],'geneName'],
					'site1_RecScore':thisGeneGroup.loc[thisGeneGroup.index[j],'Maverick_RecScore'],
					'site2_RecScore':thisGeneGroup.loc[thisGeneGroup.index[k],'Maverick_RecScore'],
					'TotalScore':harmonicMean},index=[0])],ignore_index=True)
	thisSampleFinalScores=pandas.concat([sample,compHetPairs],axis=0,sort=False,ignore_index=True)
	thisSampleFinalScores=thisSampleFinalScores.sort_values(by="TotalScore",ascending=False)
	# tidy up
	thisSampleFinalScores=thisSampleFinalScores.loc[:,['varType','hg19_chr','hg19_pos(1-based)','ref','alt','genotype','geneName','geneID','Maverick_BenignScore','Maverick_DomScore','Maverick_RecScore','varID','site1_varID','site2_varID','site1_RecScore','site2_RecScore','TotalScore']]
	thisSampleFinalScores.to_csv(base + '.finalScores.txt',sep='\t',header=True,index=False)
	return


if __name__ == "__main__":
	main(sys.argv[1:])

