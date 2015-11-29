package edu.pengli.nlp.conference.acl2015.method;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.input.SAXBuilder;

import edu.pengli.nlp.conference.acl2015.generation.AbstractiveGenerator;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;

public class ACLMethod {
	
	private static void trainingDCNN(String inputCorpusDir, 
			String outputSummaryDir, List<Element> corpusList,
			MatlabProxy proxy, int iterTime) throws 
			FileNotFoundException, IOException, 
			ClassNotFoundException, MatlabInvocationException{
		
		InstanceList patternList = new InstanceList(new Noop());
		
		for (int i = 0; i < iterTime; i++) {
			Element topic = corpusList.get(i);
			List<Element> docSets = topic.getChildren();
			Element docSetA = docSets.get(1);
			String corpusName = docSetA.getAttributeValue("id");
			
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".patterns.ser"));
			HashSet<Pattern> patternSet = (HashSet<Pattern>) in.readObject();
			for (Pattern p : patternSet) {
				Instance inst = new Instance(p, null, null, p);
				patternList.add(inst);
			}
			in.close();
		}
				
		FeatureVectorGenerator fvGenerator = new FeatureVectorGenerator();	
//		fvGenerator.trainingPatternDCNN(outputSummaryDir, patternList, proxy);
//		fvGenerator.trainingTupleDCNN(outputSummaryDir, patternList, proxy);
		fvGenerator.trainingPatternAndTupleDCNN(outputSummaryDir, patternList, proxy);
	}

	public static void main(String[] args) throws Exception {

		// Create a proxy, which we will use to control MATLAB
		String matlabLocation = "/usr/local/MATLAB/R2012a/bin/matlab";
		MatlabProxyFactoryOptions options = new MatlabProxyFactoryOptions.Builder()
				.setProxyTimeout(30000L).setMatlabLocation(matlabLocation)
				.setHidden(true).build();

		MatlabProxyFactory factory = new MatlabProxyFactory(options);
		MatlabProxy proxy = factory.getProxy();

		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();

		String outputSummaryDir = "../data/ACL2015/Output";
		String modelSummaryDir = "../data/ACL2015/ROUGE/models";
		String confFilePath = "../data/ACL2015/ROUGE/conf.xml";
		
/*		trainingDCNN(inputCorpusDir, 
				outputSummaryDir, corpusList, proxy, corpusList.size());
		System.out.println("Training is done");*/
		


		/*
		 * Pattern Generation
		 */
		/*
		 * PipeLine pipeLine = new PipeLine(); pipeLine.addPipe(new
		 * Input2CharSequence("UTF-8")); pipeLine.addPipe(new
		 * CharSequenceExtractContent( "<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		 * pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
		 * pipeLine.addPipe(new RelationExtractionbyOpenIE()); HeadAnnotation
		 * headAnnotator = new HeadAnnotation(); FramenetTagger framenetTagger =
		 * new FramenetTagger(); WordnetTagger wordnetTagger = new
		 * WordnetTagger();
		 */


		// Rouge Evaluation conf Setting
/*		ArrayList<File> files = FileOperation.travelFileList(new File(
				modelSummaryDir));
		HashMap<String, ArrayList<String>> modelSummariesMap = new HashMap<String, ArrayList<String>>();
		ArrayList<String> list = null;
		ArrayList<String> corpusNameList = new ArrayList<String>();
		for (int i = 0; i < corpusList.size(); i++) {
			System.out.println("Corpus id is " + i);
			Element topic = corpusList.get(i);
			List<Element> docSets = topic.getChildren();
			Element docSetA = docSets.get(1);
			String corpusName = docSetA.getAttributeValue("id");
			corpusNameList.add(corpusName);
		}

		for (File f : files) {
			String fn = f.getName();
			String[] toks = fn.split("\\.");
			String idx = toks[0].split("-")[0];
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
		RougeEvaluationWrapper.setConfigurationFile(corpusNameList,
				outputSummaryDir, modelSummaryDir, modelSummariesMap,
				confFilePath);*/
		
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result");

		int[] numberClusters = {6};
		int iterTime = 3;

		for (int j = 0; j < numberClusters.length; j++) {
			int numberCluster = numberClusters[j];
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
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.aclMethod2(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);

				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}

		proxy.disconnect();
		out.close();
	}

}
