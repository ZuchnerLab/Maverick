#!/bin/bash

fileName=$1

BASE=$(sed 's/\.vcf//' <<< ${fileName})

# process variants with annovar
echo "Starting Step 1: Get coding changes with Annovar"
dos2unix ${fileName}
# remove chr prefix if present
sed -i 's/^chr//' ${fileName}
grep -v '^#' ${fileName} | cut -f 1,2,4,5 > ${BASE}_locations.txt
annovar/convert2annovar.pl -format vcf4 ${BASE}.vcf > ${BASE}.avinput
annovar/annotate_variation.pl -dbtype wgEncodeGencodeBasicV33 -buildver hg38 --exonicsplicing ${BASE}.avinput annovar/humandb/
# if there are no scorable variants, end early
SCORABLEVARIANTS=$(cat ${BASE}.avinput.exonic_variant_function | wc -l || true)
if [[ ${SCORABLEVARIANTS} -eq 0 ]]; then echo "No scorable variants found"; exit 0; fi
annovar/coding_change.pl ${BASE}.avinput.exonic_variant_function annovar/humandb/hg38_wgEncodeGencodeBasicV33.txt annovar/humandb/hg38_wgEncodeGencodeBasicV33Mrna.fa --includesnp --onlyAltering --alltranscript > ${BASE}.coding_changes.txt

# select transcript
echo "Starting Step 2: Select transcript"
python Maverick/InferenceScripts/groomAnnovarOutput_GRCh38.py --inputBase=${BASE}
# if there are no scorable variants, end early
SCORABLEVARIANTS=$(cat ${BASE}.groomed.txt | wc -l || true)
if [[ ${SCORABLEVARIANTS} -lt 2 ]]; then echo "No scorable variants found"; exit 0; fi

# add on annotations
echo "Starting Step 3: Merge on annotations"
python Maverick/InferenceScripts/annotateVariants_GRCh38.py --inputBase=${BASE}

# run variants through each of the models
echo "Starting Step 4: Score variants with the 8 models"
python Maverick/InferenceScripts/runModels_GRCh38.py --inputBase=${BASE}

python Maverick/InferenceScripts/rankVariants_GRCh38.py --inputBase=${BASE}

echo "Done"
