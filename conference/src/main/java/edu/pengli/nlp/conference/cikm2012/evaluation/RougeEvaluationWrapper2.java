package edu.pengli.nlp.conference.cikm2012.evaluation;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

import edu.pengli.nlp.platform.util.TimeWait;
public class RougeEvaluationWrapper2 {

	public static HashMap run() {
		String[] cmd = { "/usr/bin/perl",
				"./models/ROUGE/ROUGE-1.5.5.pl", "-e",
				"./models/ROUGE/data", "-n", "4", "-w", "1.2", "-m",
				"-2", "4", "-u", "-c", "95", "-r", "1000", "-f", "A", "-p",
				"0.5", "-t", "0", "-a", "-d", "./data/EMNLP2012/setting_wrong.xml" };
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
//				System.out.println(line);
				if (line.contains("OurMethod ROUGE-2 Average_R:")){
					 String[] toks = line.split(" ");
					 String x = toks[2].replace("Average_", "");
					 x = x.replace(":", "");
					 map.put(x, Double.parseDouble(toks[3]));
					 
				}else if(line.contains("OurMethod ROUGE-2 Average_P:")){
					 String[] toks = line.split(" ");
					 String x = toks[2].replace("Average_", "");
					 x = x.replace(":", "");
					 map.put(x, Double.parseDouble(toks[3]));
					
				}else if(line.contains("OurMethod ROUGE-2 Average_F:")){
					 String[] toks = line.split(" ");
					 String x = toks[2].replace("Average_", "");
					 x = x.replace(":", "");
					 map.put(x, Double.parseDouble(toks[3]));
					
				}
		          
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return map;
	}


}
