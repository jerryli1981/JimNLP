package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import edu.pengli.nlp.platform.util.FileOperation;

public class parserROUGE {

	public static void main(String[] args) throws IOException {
		String dir = "../data/EMNLP2012";
		BufferedReader in = FileOperation.getBufferedReader(new File(dir),
				//"output.baseline0.n");
		      //"output.baseline1.n");
		     // "output.baseline2.cos.n");
		     //"output.baseline2.LM.n");
		     // "output.baseline3.n");
		     // "outout.ourmethod2.n");
				"output.b0.n");
			//	"output.b0.t");

		String line = null;
		double rouge = 0.0;
		while ((line = in.readLine()) != null) {
		//	System.out.println(line);
			if (line.contains("ROUGE-1 Eval")) {
				String[] toks = line.split(" ");
				System.out.println(toks[4].replace("R:", ""));
				rouge += Double.parseDouble(toks[4].replace("R:", ""));
			}
		}
		
		NumberFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);
		nf.setMinimumFractionDigits(5);
		System.out.println("Rouge-1:  "+nf.format(rouge/100));

	}

}
