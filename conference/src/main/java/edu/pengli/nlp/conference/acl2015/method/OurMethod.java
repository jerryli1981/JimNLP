package edu.pengli.nlp.conference.acl2015.method;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.json.JSONException;

import edu.pengli.nlp.conference.acl2015.generation.AbstractiveGeneration;
import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.conference.acl2015.pipe.RelationExtractionbyOpenIE;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;


public class OurMethod {

	public static void main(String[] args) throws Exception {

		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();
		ArrayList<String> corpusNameList = new ArrayList<String>();
		String outputSummaryDir = "../data/ACL2015/Output";
		
		PipeLine pipeLine = new PipeLine();
/*		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
		pipeLine.addPipe(new RelationExtractionbyOpenIE());*/
		
		
		for (int i = 24; i < corpusList.size(); i++) {
			System.out.println("Corpus id is "+i);
			Element topic = corpusList.get(i);
			String categoryId = topic.getAttributeValue("category");
			List<Element> docSets = topic.getChildren();
			Element docSetA = docSets.get(1);
			String corpusName = docSetA.getAttributeValue("id");
			corpusNameList.add(corpusName);
			AbstractiveGeneration ag = new AbstractiveGeneration();
			ag.run(inputCorpusDir + "/" + topic.getAttributeValue("id"),
					outputSummaryDir, corpusName, pipeLine, categoryId);
		}

		// Rouge Evaluation
/*		String modelSummaryDir = "../data/ACL2015/ROUGE/models";
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
		String metric = "ROUGE-SU4";
		HashMap map = RougeEvaluationWrapper.runRough(confFilePath, metric);
		System.out.println(metric + " is " + (Double) map.get(metric));*/

		System.out.println("Our method is done");

	}
}
