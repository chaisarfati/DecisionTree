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
        this.rootNode = new Node();
    }

    @Override
    public double classifyInstance(Instance instance) {
        Node currentNode = rootNode;
        while (currentNode.children != null){
            currentNode = currentNode.children[(int)instance.value(currentNode.attributeIndex)];
        }
        return currentNode.returnValue;
    }

    @Override
	public void buildClassifier(Instances arg0) throws Exception {
        buildTree(rootNode, arg0, -1);
    }

    /**
     * Returns an array containing maximumHeight at index 0
     * and averageHeight at index 1
     *
     * @param instances
     * @return
     */
    public double[] maxAndAvgHeight(Instances instances){
        double[] result = new double[2];
        int max = 0;
        int sum = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            int counter = 0;
            Node currentNode = rootNode;
            while (currentNode.children != null){
                currentNode = currentNode.children[(int)instances.instance(i).value(currentNode.attributeIndex)];
                counter++;
            }
            sum += counter;
            if(counter > max){
                max = counter;
            }
        }
        result[0] = max;
        result[1] = sum/(double)instances.numInstances();
        return result;
    }



    /**
     * Build the tree considering instances as the
     * training data and operates pre-pruning using
     * the chi-square method
     *
     * @param p_value
     * @param node
     * @param instances
     * @param minDispersion
     */
    public void buildAndPrune(int p_value, Node node, Instances instances, double minDispersion){
        double dispersion = (measureWithEntropy) ? calcEntropy(probaForEachItem(instances))
                : calcGini(probaForEachItem(instances));


        if(instances.numInstances()==0){
            node.children = null;
            node.attributeIndex = -4;
            node.returnValue = node.parent.returnValue;
            return;
        }else if(dispersion == minDispersion){
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = node.parent.returnValue;
            return;
        }else if(dispersion == 0){
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = majorityTarget(instances);
            return;
        }else{
            node.returnValue = majorityTarget(instances);
            node.attributeIndex = findOptimalIndex(instances);

            double chiSquare = calcChiSquare(instances, node.attributeIndex);
            Instances[] l = clearSet(splitSet(instances, node.attributeIndex));
            double df = l.length - 1;

            // Chi-square condition of splitting
            if(chiSquare <= chiSquareTable()[p_value][(int)df]){
                node.children = null;
                node.attributeIndex = -1;
                return;
            }

            Instances[] subset = splitSet(instances, node.attributeIndex);
            node.children = new Node[subset.length];
            for (int i = 0; i < subset.length; i++) {
                node.children[i] = new Node();
                node.children[i].parent = node;
                buildAndPrune(p_value, node.children[i], subset[i], dispersion);
            }

        }
    }


    /**
     * Build the tree considering instances as the training
     * data
     *
     * @param node
     * @param instances
     * @param previousDispersion
     */
    public void buildTree(Node node, Instances instances, double previousDispersion){
        double dispersion = (measureWithEntropy) ? calcEntropy(probaForEachItem(instances))
                : calcGini(probaForEachItem(instances));

        if(dispersion == previousDispersion){
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = majorityTarget(instances);
            return;
        }else if(instances.numInstances() == 0){
            node.children = null;
            node.attributeIndex = -4;
            node.returnValue = node.parent.returnValue;
            return;
        }else if(dispersion == 0.0){
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = majorityTarget(instances);
            return;
        }else{
            node.returnValue = majorityTarget(instances);
            node.attributeIndex = findOptimalIndex(instances);
            Instances[] subset = splitSet(instances, node.attributeIndex);

            node.children = new Node[subset.length];
            for (int i = 0; i < subset.length; i++) {
                node.children[i] = new Node();
                node.children[i].parent = node;
                buildTree(node.children[i], subset[i], dispersion);
            }
        }
    }
    /**
     * Helper method of buildTree that splits the dataset of instances
     * according to the attribute at position attributeIndex and returns
     * an array of Instances representing each split set
     *
     * @param set
     * @param attributeIndex
     * @return
     */
    public Instances[] splitSet(Instances set, int attributeIndex){
        Instances[] result = new Instances[set.attribute(attributeIndex).numValues()];
        for (int i = 0; i < result.length; i++) {
            String currentValue = set.attribute(attributeIndex).value(i);
            result[i] = new Instances(set, 0);
            for (int j = 0; j < set.numInstances(); j++) {
                if(set.instance(j).stringValue(attributeIndex).equals(currentValue)){
                    result[i].add(set.instance(j));
                }
            }
        }
        return result;
    }

    /**
     * Returns an array containing the instances of set
     * that are not empty
     *
     * @param set
     * @return
     */
    public static Instances[] clearSet(Instances[] set){
        int childLength = 0;
        for (int i = 0; i < set.length; i++) {
            if(set[i].numInstances()!=0){
                childLength++;
            }
        }
        Instances[] result = new Instances[childLength];
        int j = 0;
        for (int i = 0; i < set.length; i++) {
            if(set[i].numInstances()!=0){
                result[j] = set[i];
                j++;
            }
        }
        return result;
    }


    /**
     * Prints this tree
     */
    public void printTree(){
        printTree(rootNode, "  ");
    }
    /* Actual printing recursion*/
    private void printTree(Node node, String spaces){
        if(node.parent == null){
            System.out.println("Root");
            System.out.println("Returning value: " + node.returnValue);

        }
        for (int i = 0; i < node.children.length; i++) {
            // If the splitting training set is not empty
            if(node.children[i].attributeIndex != -4) {
                System.out.println(spaces + "If attribute " + node.attributeIndex + " = " + i);
            }else{ // Else do not print it as said in piazza
                continue;
            }
            if (node.children[i].attributeIndex != -1){
                System.out.println(spaces + "Returning value: " + node.returnValue);
                printTree(node.children[i], spaces + "  ");
            }else{
                System.out.println(spaces + "  " + "Leaf. Returning value: " + node.children[i].returnValue);
            }
        }
    }


    /**
     * Returns the class value that appears the most in
     * the instances input dataset
     * @param instances
     * @return
     */
    public double majorityTarget(Instances instances){
        int[] res = new int[instances.numClasses()];
        for (int i = 0; i < instances.numInstances(); i++) {
            res[(int)instances.instance(i).classValue()]++;
        }
        return maxIndex(res);
    }
    /*
    Auxiliary function of majorityTarget
     */
    private static double maxIndex(int[] res){
        int max = res[0];
        int result = 0;
        for (int i = 1; i < res.length; i++) {
            if(res[i] > max){
                max = res[i];
                result = i;
            }
        }
        return (double)result;
    }

    /**
     * Returns the chiSquareTable
     *
     * @return
     */
    public double[][] chiSquareTable(){
        double[][] res = {
                // 0.75
                {0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438, 9.299, 10.165, 11.037, 11.912, 12.792, 13.675, 14.562, 15.452, 16.344, 17.240, 18.137, 19.037, 19.939, 20.843, 21.749, 22.657, 23.567, 24.478, 33.660, 42.942, 52.294, 61.698, 71.145, 80.625, 90.133},
                // 0.5
                {0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340, 12.340, 13.339, 14.339, 15.338, 16.338, 17.338, 18.338, 19.337, 20.337, 21.337, 22.337, 23.337, 24.337, 25.336, 26.336, 27.336, 28.336, 29.336, 39.335, 49.335, 59.335, 69.334, 79.334, 89.334, 99.334},
                // 0.25
                {1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845, 15.984, 17.117, 18.245, 19.369, 20.489, 21.605, 22.718, 23.828, 24.935, 26.039, 27.141, 28.241, 29.339, 30.435, 31.528, 32.620, 33.711, 34.800, 45.616, 56.334, 66.981, 77.577, 88.130, 98.650, 109.141},
                // 0.05
                {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296, 27.587, 28.869, 30.144, 31.410, 32.671, 33.924, 35.142, 36.415, 37.652, 38.885, 40.113, 41.337, 42.557, 43.773, 55.758, 67.505, 79.082, 90.531, 101.879, 113.145, 124.342},
                // 0.005
                {7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300, 29.819, 31.319, 32.801, 34.267, 35.718,
                                37.156, 38.582, 39.997, 41.401, 42.796, 44.181, 45.559, 46.928, 48.290, 49.645, 50.993, 52.336, 53.672, 66.766, 79.490, 91.952, 104.215, 116.321, 128.299, 140.169}};

        return res;
    }


    /**
     * Receives a set of instances and returns the frequency
     * (in percentage) of each class in the set
     * If no instances exist in the set returns null
     *
     * @param set
     * @return setOfProbabilities
     */
    private static double[] probaForEachItem(Instances set){
        int numInstances = set.numInstances();
        if(numInstances==0){
            return null;
        }

        double[] result = new double[set.numClasses()];
        for (int i = 0; i < set.numInstances(); i++) {
            result[(int)set.instance(i).value(set.classIndex())]++;
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= numInstances;
        }
        return result;
    }


    /**
     * Computes the entropy over a set of probability
     * Returns 0 if the set is null (the set of instances it
     * represents has no instance in it)
     *
     * @param probability
     * @return
     */
    public double calcEntropy(double[] probability){
        if(probability == null){
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < probability.length; i++) {
            if(probability[i] > 0) {
                sum += probability[i] * (Math.log(probability[i])/Math.log(2));
            }
        }
        double result = -1.0 * sum;
        if(result==-0.0) result=0.0;
        return result;
    }

    /**
     * Computes the Gini coefficient for a set of probability
     * Returns 0 if the set is null (the set of instances it
     * represents has no instance in it)
     *
     * @param probability
     * @return
     */
    public double calcGini(double[] probability){
        if(probability == null){
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < probability.length; i++) {
            sum += Math.pow(probability[i],2);
        }
        return 1 - sum;
    }


    /**
     * Computes the gain of splitting the dataset according to
     * the attribute at attributeIndex and returns the information
     * gain if measureWithEntropy is set to true, the Gini gain
     * otherwise
     *
     * @param set
     * @param attributeIndex
     * @param measureWithEntropy
     * @return
     */
    public double calcGain(Instances set, int attributeIndex, boolean measureWithEntropy){
        double numInstances = set.numInstances();
        double impurityBefore;
        if(measureWithEntropy){
            impurityBefore = calcEntropy(probaForEachItem(set));
        }else{
            impurityBefore = calcGini(probaForEachItem(set));
        }

        double weightedAverageAfter = 0.0;
        Instances[] subset = splitSet(set, attributeIndex);

        double weight, impurity;
        for (int i = 0; i < set.attribute(attributeIndex).numValues(); i++) {
            weight = subset[i].numInstances()/numInstances;
            if(measureWithEntropy){
                impurity = calcEntropy(probaForEachItem(subset[i]));
            }else{
                impurity = calcGini(probaForEachItem(subset[i]));
            }
            weightedAverageAfter += weight * impurity;
        }
        return impurityBefore - weightedAverageAfter;
    }



    /**
     * Returns the index of an attribute of the instances set
     * given that will best split the set according to the
     * result of the informationGain() method
     *
     * @param instances
     * @return
     */
    public int findOptimalIndex(Instances instances){
        int index = 0;
        double max = calcGain(instances, index, measureWithEntropy);
        double currentInfo;
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            currentInfo = calcGain(instances, i, measureWithEntropy);
            if(max < currentInfo){
                max = currentInfo;
                index = i;
            }
        }
        return index;
    }


    /**
     * Returns the chi-square value of splitting i_instances
     * according to the given i_attribute
     *
     * @param instances
     * @param attributeIndex
     * @return
     */
    public double calcChiSquare(Instances instances, int attributeIndex) {
        int numOfInstances = instances.numInstances();
        int Df, pf, nf;

        double py0 = 0, py1 = 0;
        double e0, e1;
        double result = 0;


        for (int i = 0; i < numOfInstances; i++) {
            if(instances.instance(i).classValue() == 0) {
                py0++;
            }else{
                py1++;
            }
        }

        py0 /= (double)numOfInstances;
        py1 /= (double)numOfInstances;

        for (int i = 0; i < instances.attribute(attributeIndex).numValues(); i++) {
            Df = 0;
            pf = 0;
            nf = 0;
            for (int j = 0; j < instances.numInstances(); j++) {
                if (instances.instance(j).value(attributeIndex) == i) {
                    Df++;
                }
                if (instances.instance(j).classValue() == 0 && instances.instance(j).value(attributeIndex) == i) {
                    pf++;
                }
                if (instances.instance(j).classValue() == 1 && instances.instance(j).value(attributeIndex) == i) {
                    nf++;
                }
            }

            e0 = Df * py0;
            e1 = Df * py1;

            if(Df!=0) {
                result += (Math.pow(pf - e0, 2) / e0) + (Math.pow(nf - e1, 2) / e1);
            }
        }
        return result;
    }

    /**
     * Computes the average error of classifying the Instances
     * set with this tree
     *
     * @param set
     * @return
     */
    public double calcAvgError(Instances set){
        double classMistakes = classificationMistakes(set);
        double numInstances = set.numInstances();
        return classMistakes/numInstances;
    }

    /**
     * Returns the number of wrong classifications
     * over the instances of the set
     *
     * @param set
     * @return
     */
    private double classificationMistakes(Instances set){
        double result = 0;
        for (int i = 0; i < set.numInstances(); i++) {
            if(classifyInstance(set.instance(i)) != set.instance(i).classValue()){
                result++;
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

    public Node getRootNode() {
        return rootNode;
    }
}
