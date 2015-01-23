package edu.pengli.nlp.conference.acl2015.method;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.json.JSONException;

import edu.pengli.nlp.conference.acl2015.generation.AbstractiveGeneration;
import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator;
import edu.pengli.nlp.conference.acl2015.pipe.RelationExtractionbyOpenIE;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;

public class OurMethod {
	
	private static void trainingDCNN(String inputCorpusDir, 
			String outputSummaryDir, List<Element> corpusList, 
			PipeLine pipeLine, MatlabProxy proxy, int iterTime) throws 
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
				
		FeatureVectorGenerator fvGenerator = 
				(FeatureVectorGenerator) pipeLine.getPipe(0);	
		
		fvGenerator.trainingDCNN(outputSummaryDir, patternList, proxy);
	}

	public static void main(String[] args) throws Exception {
		
	    //Create a proxy, which we will use to control MATLAB
		String matlabLocation = "/usr/local/MATLAB/R2012a/bin/matlab";
		MatlabProxyFactoryOptions options = new MatlabProxyFactoryOptions.Builder()
        .setProxyTimeout(30000L).setMatlabLocation(matlabLocation)
        .setHidden(true)
        .build();
		
	    MatlabProxyFactory factory = new MatlabProxyFactory(options);
	    MatlabProxy proxy = factory.getProxy();


		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();
		ArrayList<String> corpusNameList = new ArrayList<String>();
		String outputSummaryDir = "../data/ACL2015/Output";
		
		PipeLine pipeLine = new PipeLine();
		//only used in the first phase
/*		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
		pipeLine.addPipe(new RelationExtractionbyOpenIE());*/
		
		
		//only used in the third phase
		FeatureVectorGenerator fvg = new FeatureVectorGenerator();
		pipeLine.addPipe(fvg);
		
				
		//training DCNN for pattern representation, only used in the third phase
/*		System.out.println("Begin to train DCNN pattern model");
		trainingDCNN(inputCorpusDir, 
				outputSummaryDir, corpusList, 
				pipeLine, proxy, iterTime);*/
			
		String metric = "ROUGE-SU4";
		double averageMetric = 0.0;
		int iterTime = 3;
		for(int k=0; k<iterTime; k++){
			System.out.println("Iter Time is "+k);
			for (int i = 0; i < corpusList.size(); i++) {
				System.out.println("Corpus id is "+i);
				Element topic = corpusList.get(i);
				String categoryId = topic.getAttributeValue("category");
				List<Element> docSets = topic.getChildren();
				Element docSetA = docSets.get(1);
				String corpusName = docSetA.getAttributeValue("id");
				corpusNameList.add(corpusName);
				AbstractiveGeneration ag = new AbstractiveGeneration();
				ag.run(inputCorpusDir + "/" + topic.getAttributeValue("id"),
						outputSummaryDir, corpusName, pipeLine, categoryId, proxy);
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
		
		proxy.disconnect();
		System.out.println(metric + " is " + averageMetric/iterTime);
	}
}
