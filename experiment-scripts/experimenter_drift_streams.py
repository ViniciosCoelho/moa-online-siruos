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

# To change Width for drift, you can use something like "-w 400000" (default: 1000)
# Use "-Xms<size wanted>g -Xmx<size wanted>g" to change the initial and max heap size for the JVM
base_string = "java -Xms32g -Xmx32g -cp moa-experiments.jar moa.DoTask \"EvaluatePrequential -l ENSEMBLE_NAME -s (ConceptDriftStream -s (ImbalancedStream -s STREAM_NAME -c 0.5;0.5) -d (ConceptDriftStream -s (ImbalancedStream -s STREAM_NAME -c 0.7;0.3) -d (ConceptDriftStream -s (ImbalancedStream -s STREAM_NAME -c 0.8;0.2) -d (ConceptDriftStream -s (ImbalancedStream -s STREAM_NAME -c 0.9;0.1) -d (ImbalancedStream -s STREAM_NAME -c 0.995;0.005) -p 800000) -p 600000) -p 400000) -p 200000) -e (WindowClassificationPerformanceEvaluator -w 10000 -o -p -r -f) -i " + str(instance_number) + " -f 10000 -q 10000\""

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

to_replace = ["meta.", "generators.", "functions.", "trees.", "moa.", "classifiers."]

total_quantity = len(streams) * len(ensembles)
counter = 1

for stream in streams:
	for ensemble in ensembles:
		print("\n\n##### ITERATION", counter, "OF", total_quantity, "\n\n")
		sys.stdout.flush()
		counter += 1
		command = base_string.replace("ENSEMBLE_NAME", ensemble)
		command = command.replace("STREAM_NAME", stream)
		print(command)

		file_name = "ensemble=" + ensemble \
			+ "&stream=" + stream \
			+ ".csv"

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