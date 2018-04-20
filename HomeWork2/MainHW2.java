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


        // Declares two trees
        DecisionTree treeUsingEntropy = new DecisionTree(true);
        DecisionTree treeUsingGini = new DecisionTree(false);


        // Build the two trees
        treeUsingEntropy.buildClassifier(trainingCancer);
        treeUsingGini.buildClassifier(trainingCancer);


        // Computes average errors using entropy and Gini methods
        double errorEntopy = treeUsingEntropy.calcAvgError(validationCancer),
                errorGini = treeUsingGini.calcAvgError(validationCancer);

        System.out.println("Validation error using Entropy: " + errorEntopy);
        System.out.println("Validation error using Gini: " + errorGini);
        System.out.println("----------------------------------------------------");

        // Choose the coefficient that gives the lowest avgError
        boolean measureWithEntropy = (errorEntopy < errorGini) ? true : false;


        // Construct the tree and prune it according to different p-values


        DecisionTree tree = (measureWithEntropy) ? treeUsingEntropy : treeUsingGini;
        System.out.println("Decision Tree with p_value of: " + 1);
        System.out.println("The train error of the decision tree is " + tree.calcAvgError(trainingCancer));
        System.out.println("Max height on validation data: " + tree.maxAndAvgHeight(validationCancer)[0]);
        System.out.println("Average height on validation data: " + tree.maxAndAvgHeight(validationCancer)[1]);
        double currentValidationError = tree.calcAvgError(validationCancer);
        System.out.println("The validation error of the decision tree is " + currentValidationError);
        double bestValidationError = tree.calcAvgError(validationCancer), bestPVal = 1;
        System.out.println("----------------------------------------------------");

        double[] p_values = {0.75, 0.5, 0.25, 0.05, 0.005};
        int bestIndex = 0;
        for (int i = 0; i < p_values.length; i++) {
            System.out.println("Decision Tree with p_value of: " + p_values[i]);

            // Build the tree
            tree = new DecisionTree(measureWithEntropy);
            tree.buildAndPrune(i, tree.getRootNode(), trainingCancer, -1);
            double[] maxAndAvg = tree.maxAndAvgHeight(validationCancer);

            // Print out the results
            System.out.println("The train error of the decision tree is " + tree.calcAvgError(trainingCancer));
            System.out.println("Max height on validation data: " + maxAndAvg[0]);
            System.out.println("Average height on validation data: " + maxAndAvg[1]);
            currentValidationError = tree.calcAvgError(validationCancer);
            System.out.println("The validation error of the decision tree is " + currentValidationError);
            System.out.println("----------------------------------------------------");

            // Updates the best validation error
            if(currentValidationError < bestValidationError){
                bestValidationError = currentValidationError;
                bestPVal = p_values[i];
                bestIndex = i;
            }
        }


        System.out.println("Best validation error at p_value = " + bestPVal);

        DecisionTree bestTree = new DecisionTree(measureWithEntropy);
        bestTree.buildAndPrune(bestIndex, bestTree.getRootNode(), trainingCancer, -1);
        System.out.println("Test error with best tree: " + bestTree.calcAvgError(testingCancer));

    }
}
