#CONFIG='configs/emr/supervised/config001.yaml'
CONFIG='configs/emr/with_pretraining/config001.yaml'
#CONFIG2='configs/image/quilt/config001.yaml'
CONFIG2='configs/image/features/config001.yaml'

#for N in 1
for N in 1 2 3 5
do 
	echo $CONFIG $N 
	#python ensemble.py --config $CONFIG --top_n $N
	python ensemble.py --config $CONFIG $CONFIG2 --top_n $N
done
