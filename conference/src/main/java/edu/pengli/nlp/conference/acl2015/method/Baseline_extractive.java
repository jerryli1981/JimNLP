package edu.pengli.nlp.conference.acl2015.method;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.input.SAXBuilder;

import edu.pengli.nlp.conference.acl2015.generation.ExtractiveGenerator;
import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.FeatureDocFreqPipe;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;

public class Baseline_extractive {
	
	public static void main(String[] args) throws Exception {
		
		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();
		ArrayList<String> corpusNameList = new ArrayList<String>();
		String outputSummaryDir = "../data/ACL2015/Output";
		
		PipeLine pipeLine1 = new PipeLine();
/*		pipeLine1.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine1.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		pipeLine1.addPipe(new CharSequenceCoreNLPAnnotation());*/
		
		
		String metric = "ROUGE-2";
		double averageMetric = 0.0;
		int iterTime = 3;
		for(int k=0; k<iterTime; k++){
			System.out.println("Iter Time is "+k);
			for (int i = 0; i < corpusList.size(); i++) {
				System.out.println("Corpus id is "+i);
				Element topic = corpusList.get(i);
				List<Element> docSets = topic.getChildren();
				Element docSetA = docSets.get(1);
				String corpusName = docSetA.getAttributeValue("id");
				corpusNameList.add(corpusName);
				ExtractiveGenerator.run(inputCorpusDir + "/" + topic.getAttributeValue("id"),
						outputSummaryDir, corpusName, pipeLine1);
			}
			
			// Rouge Evaluation
			String modelSummaryDir = "../data/ACL2015/ROUGE/models";
			ArrayList<File> files = FileOperation.travelFileList(new File(
					modelSummaryDir));
			HashMap<String, ArrayList<String>> modelSummariesMap = new HashMap<String, ArrayList<String>>();
			ArrayList<String> list = null;
			for (File f : files) {
				String fn = f.getName();
				String[] toks = fn.split("\\.");
				String idx = toks[0].split("-")[0]; // D1101
				String abb = idx + toks[toks.length - 2] + "-"
						+ toks[0].split("-")[1];
				if (corpusNameList.contains(abb)) {

					if (!modelSummariesMap.containsKey(abb)) {
						list = new ArrayList<String>();
						list.add(fn);
					} else {
						list = modelSummariesMap.get(abb);
						list.add(fn);
					}
					modelSummariesMap.put(abb, list);

				}
			}
			String confFilePath = "../data/ACL2015/ROUGE/conf.xml";
			RougeEvaluationWrapper.setConfigurationFile(corpusNameList,
					outputSummaryDir, modelSummaryDir, modelSummariesMap,
					confFilePath);
			
			HashMap map = RougeEvaluationWrapper.runRough(confFilePath, metric);
			Double met = (Double) map.get(metric);
			averageMetric += met;
		}
		
		System.out.println(metric + " is " + averageMetric/iterTime);
	}


}
