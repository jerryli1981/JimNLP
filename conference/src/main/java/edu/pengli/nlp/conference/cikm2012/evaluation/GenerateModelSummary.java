package edu.pengli.nlp.conference.cikm2012.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.regex.Pattern;

import edu.pengli.nlp.platform.util.FileOperation;

public class GenerateModelSummary {

	public static void main(String[] args) throws IOException {
		
         String dir = "../data/EMNLP2012/Golden_Standard";
         ArrayList<File> fs = FileOperation.travelFileList(new File(dir));
         for(File f : fs){
        	String fn = f.getName();
     		PrintWriter out = FileOperation.getPrintWriter(new File(dir),fn+".model");
     		PrintWriter out_n = FileOperation.getPrintWriter(new File(dir),fn+".model.n");
     		PrintWriter out_t = FileOperation.getPrintWriter(new File(dir),fn+".model.t");
    		out.println("<html>");
    		out.println("<head><title>" + fn+ "</title></head>");
    		out.println("<body bgcolor=\"white\">");
    		
    		out_n.println("<html>");
    		out_n.println("<head><title>" + fn+ "</title></head>");
    		out_n.println("<body bgcolor=\"white\">");
    		
    		out_t.println("<html>");
    		out_t.println("<head><title>" + fn+ "</title></head>");
    		out_t.println("<body bgcolor=\"white\">");
    		
    		BufferedReader in = FileOperation.getBufferedReader(new File(dir),fn);
    		String input = null;
    		ArrayList<String> sentList = new ArrayList<String>();
    		ArrayList<String> sentList_n = new ArrayList<String>();
    		ArrayList<String> sentList_t = new ArrayList<String>();
    		while((input = in.readLine())!= null){
    			if(input.length() == 0) continue;
    			if(input.startsWith("<N>")){
    				String tmp = input.replaceAll("<.*?>", "");
    				sentList_n.add(tmp);
    				sentList.add(tmp);
    			}else if(input.startsWith("<T>")){
    				String tmp = input.replaceAll("<.*?>", "");
    				tmp = tmp.replaceAll(
    						"(https?)://[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(/\\S*)?", "");
    				sentList_t.add(tmp);
    				sentList.add(tmp);
    			}else{
    				System.out.println(fn+" "+input);
    				System.exit(0);
    			}
    			
    		}
    		int length = 0;
    		int i = 1;
    		for (String s : sentList) {
    			out.println("<a name=\"" + i + "\">[" + i + "]</a> "
    					+ "<a href=\"#" + i + "\" " + "id=" + i + ">" + s + "</a>");
    			length += s.split(" ").length;
    			i++;
    		}
    		
    		int length_n = 0;
    		i = 1;
    		for (String s : sentList_n) {
    			out_n.println("<a name=\"" + i + "\">[" + i + "]</a> "
    					+ "<a href=\"#" + i + "\" " + "id=" + i + ">" + s + "</a>");
    			length_n += s.split(" ").length;
    			i++;
    		}
    		
    		int length_t = 0;
    		i = 1;
    		for (String s : sentList_t) {
    			out_t.println("<a name=\"" + i + "\">[" + i + "]</a> "
    					+ "<a href=\"#" + i + "\" " + "id=" + i + ">" + s + "</a>");
    			length_t += s.split(" ").length;
    			i++;
    		}
    		
    		out.println("</body>");
    		out.println("</html>");
    		out.close();
    		
    		out_n.println("</body>");
    		out_n.println("</html>");
    		out_n.close();
    		
    		out_t.println("</body>");
    		out_t.println("</html>");
    		out_t.close();
    		
    		System.out.println(fn+" "+length+" "+length_n+" "+length_t);
    		
    		//Dick_Clark 274
    		//Marie_Colvin 404
    		//Mexican_Drug_War 282
    		//Obama_same_sex_marriage_donation 325
    		//Poland_rail_crash 382
    		//Russian_jet_crash 347
    		//Russian_presidential_election 365
    		//Syrian_uprising 357
    		//Yulia_Tymoshenko_hunger_strike 341
    		//features_of_ipad3 401
    	
        	
         }

	}

}
