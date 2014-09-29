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

import edu.pengli.nlp.conference.acl2015.generation.AbstractiveGeneration;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

public class OurMethod {

	private static LexicalizedParser lp;
	
	static {
		
		String[] options = { "-maxLength", "200", "-retainTmpSubcategories" };
		
		lp = LexicalizedParser
				.loadModel("../models/Stanford/lexparser/englishPCFG.ser.gz", options);
		

	};

	public static void main(String[] args) throws JDOMException, IOException {


		
		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();
		ArrayList<String> corpusNameList = new ArrayList<String>();
		String outputSummaryDir = "../data/ACL2015/Output";
		for (int i = 0; i < corpusList.size(); i++) {
			Element topic = corpusList.get(i);
			List<Element> docSets = topic.getChildren();
			Element docSetA = docSets.get(1);
			String corpusName = docSetA.getAttributeValue("id");
			corpusNameList.add(corpusName);
			AbstractiveGeneration ag = new AbstractiveGeneration(lp);
			ag.run(inputCorpusDir + "/" + topic.getAttributeValue("id"),
					outputSummaryDir, corpusName);
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
		String metric = "ROUGE-SU4";
		HashMap map = RougeEvaluationWrapper.runRough(confFilePath, metric);
		System.out.println(metric + " is " + (Double) map.get(metric));

		System.out.println("Our method is done");

	}
}
