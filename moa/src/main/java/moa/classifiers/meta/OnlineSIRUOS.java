package moa.classifiers.meta;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.*;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.List;

public class OnlineSIRUOS extends AbstractClassifier implements MultiClassClassifier {
    public ListOption ensembleClassifierOption = new ListOption(
            "ensembleClassifierOptions",
            'm',
            "Ensemble classifier algorithms.",
            new ClassOption("learner", ' ', "", Classifier.class, "meta.OzaBag"),
            new Option[] {
                    new ClassOption("", ' ', "", Classifier.class, "bayes.NaiveBayes")
            },
            ',');

    public IntOption totalGrpsOption = new IntOption("totalGrps", 'g', "The number of classifier groups.", 10, 1,
            Integer.MAX_VALUE);

    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
            "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public MultiChoiceOption votingMethodOption = new MultiChoiceOption("votingMethod", 'V',
            "Final vote method option.",
            new String[]{"Stacking", "Simple Majority"},
            new String[]{"Stacking", "Simple Majority"},
            0);

    public ClassOption metaClassifierOption = new ClassOption("metaClassifierOption", 'n', "Meta classifier algorithm.",
            Classifier.class, "meta.OzaBag");

    // Internal attributes
    private Classifier[][] ensembleGroups;
    private Classifier metaClassifier;
    private double[] classesCount;
    private InstancesHeader metaHeader;
    protected long instancesSeen;

    // UI options
    @Override
    public String getPurposeString() {
        return "Online SIRUOS classifier";
    }

    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        Option[] learnerOptions = this.ensembleClassifierOption.getList();
        int totalGrps = this.totalGrpsOption.getValue();

        this.ensembleGroups = new Classifier[totalGrps][learnerOptions.length];

        for (int i = 0; i < learnerOptions.length; i++) {
            for (int j = 0; j < totalGrps; j++) {
                monitor.setCurrentActivity("Materializing learner " + (i + 1) + " for group " + (j + 1) + "...", -1.0);
                this.ensembleGroups[j][i] = (Classifier) ((ClassOption) learnerOptions[i].copy())
                        .materializeObject(monitor, repository);

                if (monitor.taskShouldAbort()) {
                    return;
                }

                monitor.setCurrentActivity("Preparing learner " + (i + 1) + " for group " + (j + 1) + "...", -1.0);
                this.ensembleGroups[j][i].prepareForUse(monitor, repository);

                if (monitor.taskShouldAbort()) {
                    return;
                }
            }
        }

        monitor.setCurrentActivity("Preparing meta learner...", -1.0);
        this.metaClassifier = (Classifier) (this.metaClassifierOption)
                .materializeObject(monitor, repository);
        this.metaClassifier.prepareForUse(monitor, repository);

        if (monitor.taskShouldAbort()) {
            return;
        }

        super.prepareForUseImpl(monitor, repository);
    }

    @Override
    public void resetLearningImpl() {
        for (Classifier[] grpClassifiers : this.ensembleGroups) {
            for (Classifier classifier : grpClassifiers) {
                classifier.resetLearning();
            }
        }

        this.metaClassifier.resetLearning();
    }

    protected void initClassCounts(Instance instance) {
        if (this.classesCount == null) {
            this.classesCount = new double[instance.numClasses()];
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        initClassCounts(instance);
        ++this.instancesSeen;
        ++this.classesCount[(int) instance.classValue()];

        List<Integer> memberVotes = new ArrayList<>();

        for (Classifier[] grpClassifiers : this.ensembleGroups) {
            double weight = getWeight((int) instance.classValue());
            instance.setWeight(weight);

            for (Classifier classifier : grpClassifiers) {
                // The training process happens before getting the vote
                classifier.trainOnInstance(instance);

                DoubleVector vote = new DoubleVector(classifier.getVotesForInstance(instance));
                int classification = Utils.maxIndex(vote.getArrayRef());
                memberVotes.add(classification);
            }
        }

        if (this.votingMethodOption.getChosenLabel().equalsIgnoreCase("Simple Majority")) {
            return;
        }

        Instance metaInstance = createMetaInstance(memberVotes, (int) instance.classValue());
        this.metaClassifier.trainOnInstance(metaInstance);
    }

    /**
     * Calculate the class ratio for the given index and return the inverted ratio
     * as a weight.
     *
     * @return the weight.
     */
    protected double getWeight(int classIndex) {
        double weight = 1.0D;
        int majIndex = Utils.maxIndex(this.classesCount);

        // if (majIndex != classIndex) {
        //     return weight;
        // }

        double num = classesCount[classIndex];
        // double den = Utils.sum(this.classesCount) - num;
        double den = Utils.sum(this.classesCount);

        if (den > 0.0D) {
            double classRatio = num / den;
            weight = 1.0D - classRatio;
            weight = (weight < 0.0D) ? 0.1D : weight;
            weight = weight * MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
        }

        return Math.ceil(weight);
    }

    private Instance createMetaInstance(List<Integer> memberVotes, int classification) {
        Instance metaInstance = createMetaInstance(memberVotes);
        // Add the target attribute
        metaInstance.setClassValue(classification);
        return metaInstance;
    }

    private Instance createMetaInstance(List<Integer> memberVotes) {
        // Creates meta-header, if needed
        if (this.metaHeader == null) {
            createMetaHeader();
        }

        Instance inst = new DenseInstance(metaHeader.numAttributes());

        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            inst.setValue(i, memberVotes.get(i));
        }

        inst.setDataset(metaHeader);
        inst.setClassValue(Double.NaN);

        return inst;
    }

    private void createMetaHeader() {
        List<Attribute> test = new ArrayList<>();

        for (int i = 0; i < this.ensembleGroups.length; i++) {
            for (int j = 0; j < this.ensembleGroups[0].length; j++) {
                test.add(new Attribute("memberVote" + i + "-" + j));
            }
        }

        // Needed to add the Target Class column manually
        test.add(new Attribute("hit"));

        metaHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), test, 0));
        metaHeader.setClassIndex(metaHeader.numAttributes() - 1);
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        initClassCounts(instance);

        List<Integer> memberVotes = new ArrayList<>();

        for (Classifier[] grpClassifiers : this.ensembleGroups) {
            for (Classifier classifier : grpClassifiers) {
                double[] vote = classifier.getVotesForInstance(instance);
                int classification = Utils.maxIndex(vote);
                memberVotes.add(classification);
            }
        }

        double[] finalVote;

        if (this.votingMethodOption.getChosenLabel().equalsIgnoreCase("Simple Majority")) {
            finalVote = new double[instance.numClasses()];
            
            for (int vote : memberVotes) {
                finalVote[vote]++;
            }
        } else {
            Instance metaInstance = createMetaInstance(memberVotes);
            finalVote = this.metaClassifier.getVotesForInstance(metaInstance);
        }

        return finalVote;
    }

    @Override
    public boolean isRandomizable() {
        // TODO Auto-generated method stub
        return true;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub

    }
}