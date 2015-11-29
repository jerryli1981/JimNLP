package edu.pengli.nlp.conference.cikm2015.method;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.input.SAXBuilder;

import edu.pengli.nlp.conference.cikm2015.generation.AbstractiveGenerator;
import edu.pengli.nlp.conference.cikm2015.generation.ExtractiveGenerator;
import edu.pengli.nlp.conference.cikm2015.pipe.CharSequenceExtractContent;
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

		String confFilePath = "../data/ACL2015/ROUGE/conf.xml";

		PipeLine pipeLine1 = new PipeLine();
		/*
		 * pipeLine1.addPipe(new Input2CharSequence("UTF-8"));
		 * pipeLine1.addPipe(new CharSequenceExtractContent(
		 * "<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>")); pipeLine1.addPipe(new
		 * CharSequenceCoreNLPAnnotation());
		 */

		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_extractive");

		int iterTime = 1;

		double averageMetric_1 = 0.0;
		double averageMetric_2 = 0.0;
		double averageMetric_SU4 = 0.0;

		for (int k = 0; k < iterTime; k++) {
			for (int i = 0; i < corpusList.size(); i++) {
				System.out.println("Corpus id is " + i);
				Element topic = corpusList.get(i);
				List<Element> docSets = topic.getChildren();
				Element docSetA = docSets.get(1);
				String corpusName = docSetA.getAttributeValue("id");
				ExtractiveGenerator.run(
						inputCorpusDir + "/" + topic.getAttributeValue("id"),
						outputSummaryDir, corpusName, pipeLine1);
			}

			HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
					"ROUGE-1");
			Double met_1 = (Double) map_1.get("ROUGE-1");
			System.out.println("ROUGE-1" + " " + " " + met_1);
			averageMetric_1 += met_1;

			HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
					"ROUGE-2");
			Double met_2 = (Double) map_2.get("ROUGE-2");
			System.out.println("ROUGE-2" + " " + " " + met_2);
			averageMetric_2 += met_2;

			HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
					"ROUGE-SU4");
			Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
			System.out.println("ROUGE-SU4" + " " + " " + met_SU4);
			averageMetric_SU4 += met_SU4;

		}

		System.out.println("Average ROUGE-1" + " " + " : " + averageMetric_1
				/ iterTime);
		out.println("Average  ROUGE-1" + " " + " : " + averageMetric_1
				/ iterTime);

		System.out.println("Average ROUGE-2" + " " + " : " + averageMetric_2
				/ iterTime);
		out.println("Average ROUGE-2" + " " + " : " + averageMetric_2
				/ iterTime);

		System.out.println("Average ROUGE-SU4" + " " + " : "
				+ averageMetric_SU4 / iterTime);
		out.println("Average ROUGE-SU4" + " " + " : " + averageMetric_SU4
				/ iterTime);

		out.close();

	}
}
