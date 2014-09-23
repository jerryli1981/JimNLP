package edu.pengli.nlp.conference.cikm2012.io;

import edu.pengli.nlp.platform.algorithms.lda.LDAModel;
import edu.pengli.nlp.platform.algorithms.lda.TwitterLDAModel;
import edu.pengli.nlp.platform.types.Alphabet;

public class TestTopicAlignment {
	
	private static double KL_Divergence(LDAModel model, TwitterLDAModel tmodel, int t1, int t2){
		Alphabet ldaDict = model.getAlphabet();
		double[][] ldaTWD = model.getTopicWordDistribution();
		
		Alphabet twldaDict = tmodel.getAlphabet();
		double[][] twldaTWD = tmodel.getTopicWordDistribution();
		
		Alphabet dict = new Alphabet();
		Object[] entries = ldaDict.toArray();
		for(int i=0; i<entries.length; i++){
			Object entry = entries[i];
			dict.lookupIndex(entry);
		}
		entries = twldaDict.toArray();
		for(int i=0; i<entries.length; i++){
			Object entry = entries[i];
			dict.lookupIndex(entry);
		}
		
		double div1 = 0.0;
		entries = dict.toArray();
		for(int i=0; i<entries.length; i++){
			Object obj = entries[i];
			int idxT1, idxT2;
			if(ldaDict.contains(obj)){
				idxT1 = ldaDict.lookupIndex(obj);
			}else{
				idxT1 = -1;
			}
			if(twldaDict.contains(obj)){
				idxT2 = twldaDict.lookupIndex(obj);
			}else{
				idxT2 = -1;
			}
		
			if(idxT1 != -1 && idxT2 != -1){
				double tmp = ldaTWD[t1][idxT1]/twldaTWD[t2][idxT2];
				div1 += ldaTWD[t1][idxT1] * Math.log(tmp)/Math.log(2);
			}
		}
		
		double div2 = 0.0;
		entries = dict.toArray();
		for(int i=0; i<entries.length; i++){
			Object obj = entries[i];
			int idxT1, idxT2;
			if(ldaDict.contains(obj)){
				idxT1 = ldaDict.lookupIndex(obj);
			}else{
				idxT1 = -1;
			}
			if(twldaDict.contains(obj)){
				idxT2 = twldaDict.lookupIndex(obj);
			}else{
				idxT2 = -1;
			}
			if(idxT1 != -1 && idxT2 != -1){
				double tmp = twldaTWD[t2][idxT2]/ldaTWD[t1][idxT1];
				div2 += twldaTWD[t2][idxT2] * Math.log(tmp)/Math.log(2);
			}
		
		}
		
		return (div1+div2)/2;
	}
	
	public static void main(String[] args){
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
	
		//KL-Divergence similarity metrix
		double[][] topicSimilarities = new double[numTopics][numTopics];
		for(int i=0; i< numTopics; i++){
			StringBuffer sb = new StringBuffer();
			for(int j=0; j<numTopics; j++){
				double val = KL_Divergence(model, tmodel, i, j);
				topicSimilarities[i][j] = val;
				sb.append(val+" ");
			}
			System.out.println(sb.toString());
		}	
			
		System.out.println("done");
	}

}
