package edu.pengli.nlp.conference.cikm2012.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Properties;

import org.python.util.PythonInterpreter;

import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceExtractParagraph;
import edu.pengli.nlp.conference.cikm2012.types.GoogleNewsCorpus;
import edu.pengli.nlp.platform.algorithms.lda.CCTAModel;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.SentenceDetector;
import edu.pengli.nlp.platform.pipe.SentenceTokenization;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.TimeWait;

public class PreProcessingNews {
	
	public static void main(String[] args) throws IOException, InterruptedException{
		
		String[] topics = { "Marie_Colvin", "Poland_rail_crash",
				"Russian_presidential_election", "features_of_ipad3", "Syrian_uprising",
				"Dick_Clark", "Mexican_Drug_War", "Obama_same_sex_marriage_donation",
				"Russian_jet_crash", "Yulia_Tymoshenko_hunger_strike"};
		
		PythonInterpreter interp;
		Properties props = new Properties();
		// put sentence_cleaner into Jython path
		props.setProperty("python.home", "/home/peng/Develop/Tools/Jython");
		PythonInterpreter.initialize(System.getProperties(), props, args);
		interp = new PythonInterpreter();
		interp.exec("import sentence_cleaner");
		
		for (int t = 0; t < topics.length; t++) {
			String topic = topics[t];
			System.out.println(topic);

			String GoogleNewsDir = "/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Google";

			OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
					GoogleNewsDir + "/" + String.valueOf(topic));
			
			PipeLine pipeLine = new PipeLine();
			pipeLine.addPipe(new Input2CharSequence("UTF-8"));
			pipeLine.addPipe(new CharSequenceExtractParagraph());
			GoogleNewsCorpus gc = new GoogleNewsCorpus(fIter, pipeLine);
			
			for (Instance d : gc) {
				
				String path = GoogleNewsDir + "/" + topic;
				
				PrintWriter out = FileOperation.getPrintWriter(new File(path),
						d.getName().toString());
				String content = (String) d.getData();
				out.println(content);
				out.close();

				String input = path + "/" + d.getName().toString();
				String output = path + "/" + d.getName().toString() + ".Out";

				String[] cmd = {
						"/usr/bin/python",
						"/home/peng/Develop/Workspace/NLP/models/splitta/sbd.py",
						"-m",
						"/home/peng/Develop/Workspace/NLP/models/splitta/model_nb",
						"-t", input, "-o", output };

				Process p = Runtime.getRuntime().exec(cmd);
				while (p.waitFor() != 0) {
					TimeWait.waiting(100);
				}

				BufferedReader in = FileOperation.getBufferedReader(new File(
						path), d.getName().toString() + ".Out");

				PrintWriter outClean = FileOperation.getPrintWriter(new File(
						path), d.getName().toString() + ".Out2");
				String line = null;
				while ((line = in.readLine()) != null) {
					String cleanedSent = null;
					if (line.equals(""))
						continue;
					try {
						interp.exec("o=sentence_cleaner.clean_aggressive(\""
								+ line + "\")");
						cleanedSent = interp.get("o").toString();
					} catch (Exception e) {
                         cleanedSent = line;
						
					}finally{
						
						outClean.println(cleanedSent);
						//outClean.println("<P>"+cleanedSent+"</P>");
					}
					
				}
				outClean.close();
				
				File tmp = new File(input);
				tmp.delete();
				
				File tmp2 = new File(path + "/" + d.getName().toString() + ".Out");
				tmp2.delete();

			}

		}
		
	}

}
