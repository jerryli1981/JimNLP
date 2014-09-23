package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;

import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.TimeWait;

public class RougeEvaluationWrapper {
	

	public static HashMap run(int iter, String corpusType) {
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3", "Syrian_uprising",
				"Dick_Clark", "Mexican_Drug_War", "Obama_same_sex_marriage_donation",
				"Russian_jet_crash", "Yulia_Tymoshenko_hunger_strike"};
		String parentDir = "/home/peng/Develop/Workspace/NLP/data/CIKM2012/";
		PrintWriter out = FileOperation.getPrintWriter(new File(parentDir), "setting_local.xml");
		out.println("<ROUGE_EVAL version=\"1.5.5\">");
		for(String topic : topics){
			out.println("<EVAL ID=\""+topic+"\">");
			out.println("<PEER-ROOT>");
			out.println("/home/peng/Develop/Workspace/NLP/data/CIKM2012/Output/summary");
			out.println("</PEER-ROOT>");
			out.println("<MODEL-ROOT>");
			out.println("/home/peng/Develop/Workspace/NLP/data/CIKM2012/Golden_Standard");
			out.println("</MODEL-ROOT>");
			out.println("<INPUT-FORMAT TYPE=\"SPL\">");
			out.println("</INPUT-FORMAT>");
			out.println("<PEERS>");
			out.println("<P ID=\""+iter+"\">"+topic+"."+iter+"."+corpusType+"</P>");
			out.println("</PEERS>");
			out.println("<MODELS>");
			out.println("<M ID=\"g\">"+topic+".model."+corpusType.toLowerCase()+"</M>");
			out.println("</MODELS>");
			out.println("</EVAL>");
		}
		out.println("</ROUGE_EVAL>");
		out.close();
		
		String[] cmd = { "/usr/bin/perl",
				"./models/ROUGE/ROUGE-1.5.5.pl", "-e",
				"./models/ROUGE/data", "-n", "4", "-w", "1.2", "-m",
				"-2", "4", "-u", "-c", "95", "-r", "1000", "-f", "A", "-p",
				"0.5", "-t", "0", "-a", "-d", "./data/CIKM2012/setting_local.xml"};
		HashMap<String, Double> map  = new HashMap<String, Double>();

		try {
			Process proc = Runtime.getRuntime().exec(cmd);

			try {
				while (proc.waitFor() != 0) {
					TimeWait.waiting(100);
				}
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			InputStream in = proc.getInputStream();
			BufferedReader inn = new BufferedReader(new InputStreamReader(in));
	        String line = null;
			while ((line = inn.readLine()) != null) {
	    //		System.out.println(line);
				if (line.contains("ROUGE-1 Average_R:")){
					 String[] toks = line.split(" ");
					 map.put("R", Double.parseDouble(toks[3]));
					 
				}
		          
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return map;
	}
	
}
