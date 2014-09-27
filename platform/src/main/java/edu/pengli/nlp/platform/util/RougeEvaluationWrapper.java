package edu.pengli.nlp.platform.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

public class RougeEvaluationWrapper {

	public static HashMap<String, Double> runRough(String confFilePath, String metric){
		
		HashMap<String, Double> ret  = new HashMap<String, Double>();
		
		String[] cmd = { "/usr/bin/perl",
				"../models/ROUGE/ROUGE-1.5.5.pl", "-e",
				"../models/ROUGE/data", "-n", "4", "-w", "1.2", "-m",
				"-2", "4", "-u", "-c", "95", "-r", "1000", "-f", "A", "-p",
				"0.5", "-t", "0", "-a", "-d", confFilePath};
		

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
				if (line.contains(metric+" Average_R:")){
					 String[] toks = line.split(" ");
					 ret.put(metric, Double.parseDouble(toks[3]));
					 
				}  
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ret;
		
	}
	
	public static void setConfigurationFile(ArrayList<String> corpusList, String outPutSummaryDir, String modelSummaryDir,
			HashMap<String, ArrayList<String>> modelSummariesMap, String confFilePath){
		String parentDir = confFilePath.substring(0, confFilePath.lastIndexOf("/"));
		String confName = confFilePath.substring(confFilePath.lastIndexOf("/")+1, confFilePath.length());
		PrintWriter out = FileOperation.getPrintWriter(new File(parentDir), confName);
		out.println("<ROUGE_EVAL version=\"1.5.5\">");
		for(String corpusName : corpusList){
			out.println("<EVAL ID=\""+corpusName+"\">");
			out.println("<PEER-ROOT>");
			out.println(outPutSummaryDir);
			out.println("</PEER-ROOT>");
			out.println("<MODEL-ROOT>");
			out.println(modelSummaryDir);
			out.println("</MODEL-ROOT>");
			out.println("<INPUT-FORMAT TYPE=\"SPL\">");
			out.println("</INPUT-FORMAT>");
			out.println("<PEERS>");
			out.println("<P ID=\""+0+"\">"+corpusName+"</P>");
			out.println("</PEERS>");
			out.println("<MODELS>");
			ArrayList<String> modelSummaries = modelSummariesMap.get(corpusName);
			for(int i=0; i<modelSummaries.size(); i++){
				out.println("<M ID=\""+i+"\">"+modelSummaries.get(i)+"</M>");
			}
			out.println("</MODELS>");
			out.println("</EVAL>");
		}
		
		out.println("</ROUGE_EVAL>");
		out.close();
		
		
	}	
}
