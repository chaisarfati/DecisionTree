package HomeWork2;

import weka.core.Instances;

public class Test {

    public static void main(String[] args) throws Exception {
        Instances vegetation = MainHW2.loadData("vegetation_train.txt");
        System.out.println(vegetation.attribute(3).index());
        System.out.println(vegetation.instance(0).classValue());
        System.out.println(vegetation.instance(1).classValue());
        System.out.println(vegetation.instance(2).classValue());
        DecisionTree tree = new DecisionTree(true);
        tree.buildClassifier(vegetation);

        System.out.println(tree.classifyInstance(vegetation.instance(6)));
    }

}
