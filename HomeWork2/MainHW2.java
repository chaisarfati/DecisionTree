package HomeWork2;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

        DecisionTree tree = new DecisionTree(true);


            Instances[] instances = tree.splitSet(trainingCancer, 2);
        for (Instances i :
                instances) {
            System.out.println("num instances : " + i.numInstances());
            System.out.println("entropy : " + tree.calcEntropy(DecisionTree.probaForEachItem(i)));
            System.out.println("index to split in : " + tree.findOptimalIndex(i));
            System.out.println();
        }
        System.out.println("$$$$$------$$$$");
        Instances[] inss = tree.splitSet(instances[2], 3);

        for (Instances i :
                inss) {
            System.out.println("num instances : " + i.numInstances());
            System.out.println("entropy : " + tree.calcEntropy(DecisionTree.probaForEachItem(i)));
            System.out.println("index to split in : " + tree.findOptimalIndex(i));
            System.out.println();
        }

        System.out.println("******-------------*******");
        Instances[] inss2 = tree.splitSet(instances[3], 0);

        for (Instances i :
                inss2) {
            System.out.println("num instances : " + i.numInstances());
            System.out.println("entropy : " + tree.calcEntropy(DecisionTree.probaForEachItem(i)));
            System.out.println("index to split in : " + tree.findOptimalIndex(i));
            System.out.println();
        }

        tree.buildClassifier(trainingCancer);
        System.out.println(tree.rootNode.attributeIndex);
        System.out.println(tree.rootNode.children[1].attributeIndex);

    }
}
