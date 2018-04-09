package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;

}

public class DecisionTree implements Classifier {
	private Node rootNode;
    private boolean measureWithEntropy;

    public DecisionTree(boolean measureWithEntropy) {
        this.measureWithEntropy = measureWithEntropy;
    }

    @Override
	public void buildClassifier(Instances arg0) throws Exception {
        rootNode = new Node();
        rootNode.parent = null;
        buildTree(rootNode, arg0);
	}


    public void buildTree(Node node, Instances instances){

        if(calcEntropy(probaForEachItem(instances))<=0.03){
            node.children = null;
            node.attributeIndex = -1;
            return;
        }
        node.attributeIndex = findOptimalIndex(instances);
        node.children = new Node[instances.attribute(node.attributeIndex).numValues()];
        Instances[] subset = splitSet(instances, node.attributeIndex);
        for (int i = 0; i < subset.length; i++) {
            node.children[i] = new Node();
            node.children[i].parent = node;
            buildTree(node.children[i], subset[i]);
        }

    }


    @Override
	public double classifyInstance(Instance instance) {
        return 0;
    }



    public double calcEntropy(double[] probability){
        double sum = 0;
        for (int i = 0; i < probability.length; i++) {
            if(probability[i] > 0) {
                sum += probability[i] * (Math.log(probability[i]) / Math.log(2));
            }
        }
        return -sum;
    }

    public double calcGini(double[] probability){
        double sum = 0;
        for (int i = 0; i < probability.length; i++) {
            sum += Math.pow(probability[i],2);
        }
        return 1 - sum;
    }


    public double giniGain(Instances set, int index){
        double impurityBefore = calcGini(probaForEachItem(set));
        double weightedAverageAfter = 0.0;
        Instances[] subset = splitSet(set, index);
        double weight = 0;
        for (int i = 0; i < set.attribute(index).numValues(); i++) {
            weight = subset[i].numInstances()/(double)set.numInstances();
            weightedAverageAfter += weight * calcGini(probaForEachItem(subset[i]));
        }
        return impurityBefore - weightedAverageAfter;
    }

    public double informationGain(Instances set, int index){
        double impurityBefore = calcEntropy(probaForEachItem(set));
        double weightedAverageAfter = 0.0;
        Instances[] subset = splitSet(set, index);

        double weight = 0;
        for (int i = 0; i < set.attribute(index).numValues(); i++) {
            weight = subset[i].numInstances()/(double)set.numInstances();
            weightedAverageAfter += weight * calcEntropy(probaForEachItem(subset[i]));
        }
        return impurityBefore - weightedAverageAfter;
    }


    public static double[] probaForEachItem(Instances set){
        if(set.numInstances()==0){
            return new double[2];
        }
        double fromClassA = 0, fromClassB = 0;
        for (int i = 0; i < set.numInstances(); i++) {
            if(set.instance(i).stringValue(set.classIndex()).equals("recurrence-events")){
                fromClassA++;
            }else{
                fromClassB++;
            }
        }
        double[] result = new double[set.numClasses()];
        result[0] = fromClassA/(double)set.numInstances();
        result[1] = fromClassB/(double)set.numInstances();
        return result;
    }


    public int findOptimalIndex(Instances instances){
        int index = 0;
        double max = informationGain(instances, index);
        double currentInfo = 0;
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            currentInfo = informationGain(instances, i);
            if(max < currentInfo){
                max = currentInfo;
                index = i;
            }
        }
        return index;
    }

    public Instances[] splitSet(Instances set, int attributeIndex){
        Instances[] result = new Instances[set.attribute(attributeIndex).numValues()];
        for (int i = 0; i < result.length; i++) {
            String currentValue = set.attribute(attributeIndex).value(i);
            result[i] = new Instances(set, set.numInstances());
            for (int j = 0; j < set.numInstances(); j++) {
                if(set.instance(j).stringValue(attributeIndex).equals(currentValue)){
                    result[i].add(set.instance(j));
                }
            }
        }
        return result;
    }


    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
