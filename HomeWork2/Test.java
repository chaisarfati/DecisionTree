package HomeWork2;

import weka.core.Attribute;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;

public class Test {

    public static void main(String[] args) throws IOException {
        Instances vegetation = MainHW2.loadData("vegetation_train.txt");
        ArrayList<Attribute> list = new ArrayList<>();
        for (int i = 0; i < vegetation.numAttributes()-1; i++) {
            list.add(vegetation.attribute(i));
        }
        DecisionTree tree = new DecisionTree(true);
        tree.rootNode = new Node();
        tree.id32(tree.rootNode, vegetation, list);
        System.out.println(tree.);
    }

}
