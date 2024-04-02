import os, sys

def divide(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
			out.append(seq[int(last):int(last + avg)])
			last += avg
	return out

# NODE NUMBERS:
# Already used: All
total_nodes = 4
node_number = 0

if node_number > total_nodes - 1:
	raise ValueError("Node number must be between 0 and " + str(total_nodes - 1))

instance_number = 1000000

# Use "-Xms<size wanted>g -Xmx<size wanted>g" to change the initial and max heap size for the JVM
base_string = "java -Xms32g -Xmx32g -cp moa-experiments.jar moa.DoTask \"EvaluatePrequential -l ENSEMBLE_NAME -s (ImbalancedStream -s STREAM_NAME -c IMB_RATIO) -e (WindowClassificationPerformanceEvaluator -w 10000 -o -p -r -f) -i " + str(instance_number) + " -f 10000 -q 10000\""

streams = [
	"generators.AgrawalGenerator",
	"generators.SEAGenerator",
	"generators.AssetNegotiationGenerator",
	"(generators.RandomTreeGenerator -r 1 -i 1)"
]

streams = divide(streams, total_nodes)
streams = streams[node_number]

n_cpu = -1

ensembles = [
	# "bayes.NaiveBayes",
	# "trees.HoeffdingTree",
	# "(meta.AdaptiveRandomForest -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	# "(meta.AdaptiveRandomForestRE -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	# "(meta.KUE -n 100)",
	# "(meta.OzaBag -s 100)",
	# "(meta.OzaBoost -s 100)",
	# "(meta.OzaBagAdwin -s 100)",
	# "(meta.LeveragingBag -s 100)",
	# "(meta.LearnNSE -e 100)",
	# "(meta.CSARF -j " + str(n_cpu) + " -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01))",
	# "(meta.OnlineAdaC2 -l trees.HoeffdingTree -s 100)",
	# "(meta.OnlineCSB2 -l trees.HoeffdingTree -s 100)",
	# "(meta.OnlineUnderOverBagging -l trees.HoeffdingTree -s 100)"

	"(meta.OnlineSIRUOS -s 100 -n (meta.AdaptiveRandomForest -s 100))",
	"(meta.OnlineSIRUOS -s 100 -n (meta.OzaBag -s 100))",
	"(meta.OnlineSIRUOS -s 100 -n bayes.NaiveBayes)"
]

imb_ratios = [
	"0.5;0.5",
	"0.7;0.3",
	"0.8;0.2",
	"0.9;0.1",
	"0.95;0.05",
	"0.99;0.01",
	"0.995;0.005"
]

to_replace = ["meta.", "generators.", "functions.", "trees.", "moa.", "classifiers."]

total_quantity = len(streams) * len(ensembles) * len(imb_ratios)
counter = 1

for stream in streams:
	for ensemble in ensembles:
		for imb_ratio in imb_ratios:
			print("\n\n##### ITERATION", counter, "OF", total_quantity, "\n\n")
			sys.stdout.flush()
			counter += 1
			command = base_string.replace("ENSEMBLE_NAME", ensemble)
			command = command.replace("STREAM_NAME", stream)
			command = command.replace("IMB_RATIO", imb_ratio)
			print(command)

			file_name = "ensemble=" + ensemble + "&stream=" + stream + "&imb_ratio=" + imb_ratio + ".csv"

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