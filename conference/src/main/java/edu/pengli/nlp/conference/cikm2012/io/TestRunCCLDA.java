package edu.pengli.nlp.conference.cikm2012.io;

import java.util.ArrayList;

import edu.pengli.nlp.platform.algorithms.lda.CCLDAModel;
import edu.pengli.nlp.platform.algorithms.lda.LDAModel;
import edu.pengli.nlp.platform.algorithms.lda.TwitterLDAModel;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;

public class TestRunCCLDA {

	public static void main(String[] args) {
		double alpha = 10; // 50/numTopics
		double beta = 0.01;
		double gamma = 20;
		int numTopics = 5;
		int numIters = 100;


		LDAModel model = new LDAModel(numTopics, alpha, beta, numIters);
		
		String outputDir = "/home/peng/Develop/Workspace/NLP/data/EMNLP2012/";
		String outputName = "newsTopicsGoogle";
		model.readModel(outputDir, outputName);
		
		TwitterLDAModel tmodel = new TwitterLDAModel(numTopics, alpha, beta,
				gamma, numIters);
		tmodel.readModel(outputDir, "newsTopicsTwitter");
		
		InstanceList list1 = model.getInstanceList();
		InstanceList list2 = tmodel.getInstanceList();
		InstanceList list_t = new InstanceList(null);
		for(Instance inst : list2){
			InstanceList user = (InstanceList) inst.getData();
			for(Instance tweet : user){
				list_t.add(tweet);
			}
		}
		list_t.setDataAlphabet(list2.getDataAlphabet());

		ArrayList<InstanceList> colls = new ArrayList<InstanceList>();
		colls.add(list1);
		colls.add(list_t);
		
		alpha = 10;
		beta = 0.01;
		double gamma0 = 1.0;
		double gamma1 = 1.0;
		numTopics = 5;
		numIters = 100;
		
		CCLDAModel ccmodel = new CCLDAModel(numTopics, alpha, beta,
				gamma0, gamma1, numIters);
		
		ccmodel.initEstimate(colls);
		ccmodel.estimate();
		ccmodel.output_model();
		
		System.out.println("done");
		
	}

}
