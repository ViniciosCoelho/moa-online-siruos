import os, sys

def divide(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
			out.append(seq[int(last):int(last + avg)])
			last += avg
	return out

def get_file_num_instances(file_path: str):
	num_instances = -1

	try:
		file = open(file_path, 'r')
		lines = file.readlines()
		found_data_start = False
		data_start_line_num = -1

		for line_num, line in enumerate(lines):
			formatted_line = line.lower().strip()

			if formatted_line == '@data':
				found_data_start = True
				data_start_line_num = line_num
				break
		
		if not found_data_start:
			print('File without ARFF format! Ignoring \"' + file_path + '\"...')

		filtered_lines = lines[data_start_line_num + 1:]
		num_instances = 0

		for line in filtered_lines:
			formatted_line = line.lower().strip()

			if line == '\n' or line == '' or line.startswith('%'):
				continue

			num_instances += 1
	except FileNotFoundError:
		print('\tError - \"' + file_path + '\" couldn\'t be open...')
	
	return num_instances

# NODE NUMBERS:
# Already used: All
total_nodes = 1
node_number = 0

if node_number > total_nodes - 1:
	raise ValueError("Node number must be between 0 and " + str(total_nodes - 1))

# Use "-Xms<size wanted>g -Xmx<size wanted>g" to change the initial and max heap size for the JVM
base_string = "java -Xms8g -Xmx8g -cp moa-experiments.jar moa.DoTask \"EvaluatePrequential -l ENSEMBLE_NAME -s (ArffFileStream -f STREAM_NAME -c CLASS_INDEX) -e (WindowClassificationPerformanceEvaluator -w WIDTH -o -p -r -f) -f SAMPLE_FREQ -q SAMPLE_FREQ\""

streams = [
	(17    , "streams/bank-marketing.arff"),
	(14    , "streams/bike_sharing.arff"),
	(13    , "streams/BNG_bridges-1vsAll.arff"),
	(13    , "streams/BNG_bridges.arff"),
	(20    , "streams/BNG_hepatitis.arff"),
	(19    , "streams/BNG_lymph.arff"),
	(14    , "streams/BNG_wine.arff"),
	(18    , "streams/BNG_zoo.arff"),
	(8     , "streams/bpam.arff"),
	(10    , "streams/breast.cancer.arff"),
	(42    , "streams/census.arff"),
	(7     , "streams/chess.arff"),
	(43    , "streams/connect-4.arff"),
	(55    , "streams/covertype.arff"),
	(55    , "streams/covertypeSorted.arff"),
	(73    , "streams/covPokElec.arff"),
	(55    , "streams/covtypeNorm-1-2vsAll.arff"),
	(55    , "streams/covtypeNorm.arff"),
	(31    , "streams/creditcardfraud.arff"),
	(177   , "streams/CSDS1.arff"),
	(36    , "streams/CSDS2.arff"),
	(151   , "streams/CSDS3.arff"),
	(4     , "streams/dataset-colors-rgb-hsl-hsv-CD.arff"),
	(4     , "streams/dataset-rgbcolorsCD.arff"),
	(24    , "streams/earthquake.arff"),
	(4     , "streams/export_rgb_hsl_CD_optimized.arff"),
	(11    , "streams/giveMeLoanKaggle.arff"),
	(6     , "streams/IntelLabSensors-1-2-3-4-5-6-7-8-9vsAll.arff"),
	(6     , "streams/IntelLabSensors-1-2-3-4-5vsAll.arff"),
	(6     , "streams/IntelLabSensors-1-2-3vsAll.arff"),
	(6     , "streams/IntelLabSensors-1vsAll.arff"),
	(6     , "streams/IntelLabSensors.arff"),
	(1559  , "streams/internet_ads.arff"),
	(481   , "streams/kdd98.arff"),
	(42    , "streams/kdd99.arff"),
	(42    , "streams/kdd99_binary.arff"),
	(15    , "streams/kyoto.arff"),
	(17    , "streams/letterRecognition.arff"),
	(1     , "streams/lung-cancer.arff"),
	(119   , "streams/nomao.arff"),
	(22    , "streams/outdoor.arff"),
	(11    , "streams/poker-lsn-1-2vsAll.arff"),
	(11    , "streams/poker-lsn.arff"),
	(11    , "streams/pokerhand.arff"),
	(11    , "streams/pokerhand1M.arff"),
	(3     , "streams/powersupply.arff"),
	(31    , "streams/pozzolo_credit_card.arff"),
	(18    , "streams/primary.tumor.arff"),
	(28    , "streams/rialto.arff"),
	(371   , "streams/santander.arff"),
	(6     , "streams/sensor.arff"),
	(6     , "streams/sensorStream.arff"),
	(10    , "streams/shuttle.arff"),
	(10    , "streams/shuttle_tst_trn_comma_separeted.arff"),
	(30    , "streams/sick.arff"),
	(10883 , "streams/spam"),
	(500   , "streams/spam_corpus.arff"),
	(123   , "streams/speeddating.arff"),
	(12    , "streams/wineRed.arff"),
	(12    , "streams/wineRed100instancias.arff"),
	(12    , "streams/wineWhite.arff")
]

streams = divide(streams, total_nodes)
streams = streams[node_number]

n_cpu = -1

ensembles = [
	"bayes.NaiveBayes",
	"trees.HoeffdingTree",
	"(meta.AdaptiveRandomForest -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	"(meta.AdaptiveRandomForestRE -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	"(meta.KUE -n 100)",
	"(meta.OzaBag -s 100)",
	"(meta.OzaBoost -s 100)",
	"(meta.OzaBagAdwin -s 100)",
	"(meta.LeveragingBag -s 100)",
	"(meta.LearnNSE -e 100)",
	"(meta.CSARF -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	"(meta.OnlineAdaC2 -l trees.HoeffdingTree -s 100)",
	"(meta.OnlineCSB2 -l trees.HoeffdingTree -s 100)",
	"(meta.OnlineUnderOverBagging -l trees.HoeffdingTree -s 100)"
]

to_replace = ["meta.", "generators.", "functions.", "trees.", "moa.", "classifiers."]

total_quantity = len(streams) * len(ensembles)
counter = 1

for (class_index, stream) in streams:
	num_instances = get_file_num_instances(stream)

	if num_instances == -1:
		continue

	sample_freq = num_instances // 10
	width = sample_freq

	for ensemble in ensembles:
		print("\n\n##### ITERATION", counter, "OF", total_quantity, "\n\n")
		sys.stdout.flush()
		counter += 1
		command = base_string.replace("ENSEMBLE_NAME", ensemble)
		command = command.replace("STREAM_NAME", stream)
		command = command.replace("WIDTH", str(width))
		command = command.replace("SAMPLE_FREQ", str(sample_freq))
		command = command.replace("CLASS_INDEX", str(class_index))
		print(command)

		file_name = "ensemble=" + ensemble + "&file=" + stream.split('/')[1] + "&class_index=" + str(class_index) + ".csv"

		for i in to_replace:
			file_name = file_name.replace(i,"")

		output_file = "./results/" + file_name

		if os.path.isfile(output_file):
			if os.path.getsize(output_file) > 0:
				print('Experiment already done. Moving to next one.')
				continue

		command += " > \"" + output_file + "\""
		os.system(command)
		print('Output file: "' + output_file + '"')