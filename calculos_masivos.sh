#!/bin/bash
echo Fecha: `date` > output_files/training_LPTRNOOT/2020-05-all.txt
echo ============================== >>  output_files/training_LPTRNOOT/2020-05-all.txt
echo WITH PCA >>  output_files/training_LPTRNOOT/2020-05-all.txt
 
# Cargar modulo
module load Anaconda3/

# Activar entorno de conda
source activate AAA3

# Inicia entrenamiento
python3 USECASE1_CONTINUOUS_CLASSIFICATION.py training > output_files/training_LPTRNOOT/2020-05-t.txt

# Inicia analyzing
python3 USECASE1_CONTINUOUS_CLASSIFICATION.py analyzing > output_files/training_LPTRNOOT/2020-05-a.txt

# Inicia making_decision
python3 USECASE1_CONTINUOUS_CLASSIFICATION.py making_decision > output_files/training_LPTRNOOT/2020-05-md.txt

# Inicia tag-files-per-class
# OJO: CAMBIAR DIAS!
# OJO: Checar que la ruta de los resultados (res) sea la correcta!!
python3 tag_file_per_class.py /home/calo/compartido/AAA-master/LPTRNOOT/res/LPTRNOOT/res 2020 122 152 > output_files/training_LPTRNOOT/2020-05-tf.txt

# Desactivar entorno
conda deactivate

# Tiempo final
echo Fecha de final: `date` >>  output_files/training_LPTRNOOT/2020-05-all.txt

# Termina trabajo
echo ============================== >>  output_files/training_LPTRNOOT/2020-05-all.txt

