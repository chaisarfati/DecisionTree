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

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

	}
    
    @Override
	public double classifyInstance(Instance instance) {
        return 0;
    }


    public double calcEntropy(double[] probability){
        double sum = 0;
        for (int i = 0; i < probability.length; i++) {
            sum += probability[i] * (Math.log(probability[i])/Math.log(2));
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
