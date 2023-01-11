#!/bin/bash

genome="grch37"

usage="$(basename "$0") [option] input.vcf -- Maverick: a Mendelian approach to variant effect prediction

Usage:
	-h	show this help message
	-g	input VCF file uses GRCh38 instead of the default of GRCh37"

while getopts ":hg" optionName; do
	case "$optionName" in
	h)	echo "$usage"
		exit
		;;
	g)	genome="grch38"
		;;
	\?)	printf "illegal option: -%s\n" "$OPTARG" >&2
		echo "$usage" >&2
		exit 1
		;;
	esac
done

shift $((OPTIND - 1))

if [ -z "$1" ]; then echo "$usage"; exit; fi

fileName=$1

if [[ "$genome" == "grch37" ]]; then bash Maverick/InferenceScripts/scoreVariants.sh ${fileName}
else bash Maverick/InferenceScripts/scoreVariants_GRCh38.sh ${fileName}
fi
