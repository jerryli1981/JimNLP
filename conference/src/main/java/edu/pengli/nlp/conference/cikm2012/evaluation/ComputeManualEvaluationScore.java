package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class ComputeManualEvaluationScore {


	public static void main(String[] args) throws IOException {
		String path = "/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Output/summary";
		String methodName = "Ours";
		File dir = new File(path+"/"+methodName);
		File[] fs = dir.listFiles();
		int count = 0;
		double score = 0.0;
		for(File f : fs){
			BufferedReader in = new BufferedReader(new FileReader(f));
			String input = null;
			int lc = 0;
			double ls = 0.0;
			while((input = in.readLine()) != null){
				if(input.startsWith("<SP")){
					String mention = input.replaceAll("<|>", "");
					mention = mention.replace("SP score=", "");
					mention = mention.trim();
					score += Double.parseDouble(mention);
					ls +=Double.parseDouble(mention);
					count++;
					lc ++;
				}
			}
			System.out.println(f.getName() + ls/lc);
		}
		System.out.println(score/count);

	}

}
