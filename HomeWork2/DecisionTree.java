package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
    double attributeValue;
	double returnValue;
    Instances set;
}

public class DecisionTree implements Classifier {
	public Node rootNode;
    private boolean measureWithEntropy;

    public DecisionTree(boolean measureWithEntropy) {
        this.measureWithEntropy = measureWithEntropy;
    }

    @Override
	public void buildClassifier(Instances arg0) throws Exception {
        rootNode = new Node();
        rootNode.set = arg0;
        rootNode.parent = null;
        //buildTree(rootNode, arg0);
    }


    public void buildTree2(){
        ConcurrentLinkedQueue<Node> queue = new ConcurrentLinkedQueue<Node>();
        queue.offer(rootNode);
        while (!queue.isEmpty()){
            System.out.println(queue.size());
            Node n = queue.poll();
            if(calcEntropy(probaForEachItem(n.set))==0){
                n.children = null;
                n.attributeIndex = -1;
                n.returnValue = majorityTarget(n.set);
                System.out.println("perfectly classified");
            }else{
                System.out.println("not perfect");
                n.attributeIndex = findOptimalIndex(n.set, null);
                n.returnValue = majorityTarget(n.set);
                Instances[] subset = clearSet(splitSet(n.set, n.attributeIndex));
                n.children = new Node[subset.length];
                for (int i = 0; i < n.children.length; i++) {
                    n.children[i] = new Node();
                    n.children[i].parent = n;
                    n.children[i].set = subset[i];
                    queue.offer(n.children[i]);
                }
            }
        }
    }


    public void buildTree(Node node, Instances instances, List<Attribute> att){
        if(att.isEmpty()){
            System.out.println("no more features");
            node.children = null;
            node.attributeIndex = -1;
            return;
        } else if(calcEntropy(probaForEachItem(instances))==0){
            System.out.println("pure");
            node.children = null;
            node.attributeIndex = -1;
            return;
        }
        System.out.println("more work");
        node.attributeIndex = findOptimalIndex(instances, att);
        att.remove(instances.attribute(node.attributeIndex));

        Instances[] subset = splitSet(instances, node.attributeIndex);
        node.children = new Node[subset.length];
        for (int i = 0; i < node.children.length; i++) {
            node.children[i] = new Node();
            node.children[i].parent = node;
            buildTree(node.children[i], subset[i], att);
        }
    }

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

    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }


    public double classify(Instance instance, Node n) {
        if(n.children==null){
            return n.returnValue;
        }

        for (int i = 0; i < n.children.length; i++) {
            if(n.children[i].attributeValue==instance.value(n.attributeIndex)){
                classify(instance, n.children[i]);
            }
        }
        return 0;
    }


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


    public double giniGain(Instances set, int attributeIndex){
        double impurityBefore = calcGini(probaForEachItem(set));
        double weightedAverageAfter = 0.0;
        Instances[] subset = splitSet(set, attributeIndex);
        double weight = 0;
        for (int i = 0; i < set.attribute(attributeIndex).numValues(); i++) {
            weight = subset[i].numInstances()/(double)set.numInstances();
            weightedAverageAfter += weight * calcGini(probaForEachItem(subset[i]));
        }
        return impurityBefore - weightedAverageAfter;
    }

    public double informationGain(Instances set, int attributeIndex){
        double numInstances = set.numInstances();
        double impurityBefore = calcEntropy(probaForEachItem(set));
        double weightedAverageAfter = 0.0;
        Instances[] subset = splitSet(set, attributeIndex);

        double weight = 0;
        for (int i = 0; i < set.attribute(attributeIndex).numValues(); i++) {
            weight = subset[i].numInstances()/numInstances;
            weightedAverageAfter += weight * calcEntropy(probaForEachItem(subset[i]));
        }
        return impurityBefore - weightedAverageAfter;
    }


    public static double[] probaForEachItem(Instances set){
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


    public int findOptimalIndex(Instances instances, List<Attribute> attributes){
        int index = attributes.get(0).index();
        double max = informationGain(instances, index);
        double currentInfo;
        for (int i = 1; i < attributes.size(); i++) {
            currentInfo = informationGain(instances, attributes.get(i).index());
            if(max < currentInfo){
                max = currentInfo;
                index = attributes.get(i).index();
            }
        }
        return index;
    }

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


    public void id3(Node node, Instances instances, List<Attribute> attributes){

        if(instances.numInstances()==0){
            System.out.println("empty dataset");
            return;
        }else if(calcEntropy(probaForEachItem(instances))== 0.0 && instances.numInstances()!=0){
            System.out.println("completely pure");
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = instances.instance(0).classValue();
            return;
        }else if(attributes.isEmpty()){
            System.out.println("no more features");
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = majorityTarget(instances);
            return;
        }else{
            System.out.println("there is more work");
            node.returnValue = majorityTarget(instances);
            node.attributeIndex = findOptimalIndex(instances, attributes);
            attributes.remove(instances.attribute(node.attributeIndex));
            Instances[] subset = splitSet(instances, node.attributeIndex);
            node.children = new Node[subset.length];
            for (int i = 0; i < subset.length; i++) {
                node.children[i] = new Node();
                node.children[i].parent = node;
                id3(node.children[i], subset[i], attributes);
            }
        }
    }


    public double majorityTarget(Instances instances){
        int[] res = new int[instances.numClasses()];
        for (int i = 0; i < instances.numInstances(); i++) {
            res[(int)instances.instance(i).classValue()]++;
        }
        return maxIndex(res);
    }

    public static double maxIndex(int[] res){
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



    public void id32(Node node, Instances instances, List<Attribute> attributes){

        if(calcEntropy(probaForEachItem(instances))==0 && instances.numInstances()!=0){
            System.out.println("pure!!!");
            return;
        }else if(attributes.isEmpty()){
            System.out.println("attributes empty");
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = instances.instance(0).classValue();
            return;
        }else if(instances.numInstances()==0){
            System.out.println("empty dataset");
            node.children = null;
            node.attributeIndex = -1;
            node.returnValue = majorityTarget(instances);
            return;
        }else{
            System.out.println("there is more work");
            node.returnValue = majorityTarget(instances);
            node.attributeIndex = findOptimalIndex(instances, attributes);
            Instances[] subset = splitSet(instances, node.attributeIndex);
            attributes.remove(instances.attribute(node.attributeIndex));
            node.children = new Node[subset.length];
            for (int i = 0; i < subset.length; i++) {
                node.children[i] = new Node();
                node.children[i].parent = node;
                id3(node.children[i], subset[i], attributes);
            }
        }
    }

}
