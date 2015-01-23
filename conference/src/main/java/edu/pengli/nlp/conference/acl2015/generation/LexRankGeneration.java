package edu.pengli.nlp.conference.acl2015.generation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.platform.algorithms.ranking.LexRank;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.FeatureDocFreqPipe;
import edu.pengli.nlp.platform.pipe.FeatureSequence2FeatureVector;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureCounter;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Summary;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class LexRankGeneration {

	// cosine similarity
	private static double getRedundancyScore(Instance candidate,
			Summary currentSummary) {
		double rs = 0.0;
		FeatureVector[] tf_idf_fv_can = (FeatureVector[]) candidate.getData();
		FeatureVector tf_fv_can = tf_idf_fv_can[0];
		FeatureVector idf_fv_can = tf_idf_fv_can[1];
		int[] index = tf_fv_can.getIndices();
		int[] INDEX = new int[index.length];
		double[] VAL = new double[index.length];
		for (int i = 0; i < index.length; i++) {
			INDEX[i] = index[i];
			VAL[i] = tf_fv_can.getValues()[i] * idf_fv_can.getValues()[i];
		}
		FeatureVector fv_can = new FeatureVector(INDEX, VAL);

		ArrayList<Integer> idxList = new ArrayList<Integer>();
		ArrayList<Double> valList = new ArrayList<Double>();
		for (Instance inst : currentSummary) {
			FeatureVector[] tf_idf_fv = (FeatureVector[]) inst.getData();
			FeatureVector tf_fv = tf_idf_fv[0];
			FeatureVector idf_fv = tf_idf_fv[1];
			for (int i = 0; i < tf_fv.getIndices().length; i++) {
				idxList.add(tf_fv.getIndices()[i]);
				double v = tf_fv.getValues()[i] * idf_fv.getValues()[i];
				valList.add(v);
			}
		}
		INDEX = new int[idxList.size()];
		VAL = new double[idxList.size()];
		for (int i = 0; i < INDEX.length; i++) {
			INDEX[i] = idxList.get(i);
			VAL[i] = valList.get(i);
		}
		FeatureVector fv_sum = new FeatureVector(INDEX, VAL);

		ArrayList<Integer> commonIdx = new ArrayList<Integer>();

		int[] index_c = fv_can.getIndices();
		double[] val_c = fv_can.getValues();
		int[] index_s = fv_sum.getIndices();
		double[] val_s = fv_sum.getValues();
		for (int i = 0; i < index_c.length; i++) {
			int x = index_c[i];
			for (int j = 0; j < index_s.length; j++) {
				int y = index_s[j];
				if (x == y)
					commonIdx.add(x);
			}
		}

		double sum = 0;
		for (int i = 0; i < commonIdx.size(); i++) {
			int idx = commonIdx.get(i);
			sum += val_c[fv_can.location(idx)] * val_s[fv_sum.location(idx)];
		}

		rs = sum / (fv_can.twoNorm() * fv_sum.twoNorm());
		return rs;
	}

	private static int getHighestPosition(ArrayList<Double> score) {
		double max = Double.NEGATIVE_INFINITY;
		int position = -1;

		if (score.size() == 0) {
			return -1;
		}
		for (int i = 0; i < score.size(); i++) {
			if (score.get(i) >= max) {
				max = score.get(i);
				position = i;
			}
		}
		try {
			score.set(position, Double.NEGATIVE_INFINITY);
		} catch (Exception e) {
			System.out.println();
		}

		return position;
	}
	
	

	public static void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine) throws FileNotFoundException, IOException, ClassNotFoundException {

		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);
		PipeLine docFreqPipeLine = new PipeLine();
		docFreqPipeLine.addPipe(new Input2CharSequence("UTF-8"));
		docFreqPipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		docFreqPipeLine.addPipe(new CharSequence2TokenSequence());
		docFreqPipeLine.addPipe(new TokenSequenceLowercase());
		docFreqPipeLine.addPipe(new TokenSequenceRemoveStopwords());
		docFreqPipeLine.addPipe(new TokenSequence2FeatureSequence());
		InstanceList corpus = new InstanceList(docFreqPipeLine);
		corpus.addThruPipe(fIter);
		Alphabet dict = corpus.getDataAlphabet();
		
		FeatureDocFreqPipe dfCounterPipe = new FeatureDocFreqPipe(dict, null);
		docFreqPipeLine = new PipeLine();
		docFreqPipeLine.addPipe(dfCounterPipe);
		InstanceList corpus2 = new InstanceList(docFreqPipeLine);
		corpus2.addThruPipe(corpus.iterator());
		FeatureCounter dfCounter = dfCounterPipe.getFeatureDocCounter();
	
/*		fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);
		InstanceList docs = new InstanceList(pipeLine);
		docs.addThruPipe(fIter);
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				 outputSummaryDir + "/" +corpusName + ".ser.extractive")); 
		docs.writeObject(out);
	    out.close();*/
		
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser.extractive"));
		InstanceList docs = new InstanceList(null);
		docs.readObject(in);
		in.close();
		
		InstanceList totalSentenceList = new InstanceList(null);
		for (Instance doc : docs) {
			Annotation document = (Annotation) doc.getData();
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap cm :sentences){
				String mention = cm.toString();
				Instance sent = new Instance(mention, null, null, mention);
				totalSentenceList.add(sent);
			}
		}
		
		pipeLine = new PipeLine();
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		pipeLine.addPipe(new FeatureSequence2FeatureVector());
		InstanceList tf_fvs = new InstanceList(pipeLine);
		tf_fvs.addThruPipe(totalSentenceList.iterator());
		

		pipeLine = new PipeLine();
		InstanceList tf_idf_fvs = new InstanceList(pipeLine);
		int numDocs = docs.size();
		for(Instance inst : tf_fvs){
			
			FeatureVector tf_fv = (FeatureVector) inst.getData();
			int[] indexs = tf_fv.getIndices();
			double[] vals = new double[indexs.length];
			for (int i = 0; i < indexs.length; i++) {
				int idx = indexs[i];
				double df = dfCounter.get(idx);
				double idf = Math.log10(numDocs / (df));
				vals[i] = idf;
			}

			FeatureVector idf_fv = new FeatureVector(indexs, vals);
			FeatureVector[] tf_idf_fv = new FeatureVector[2];
			tf_idf_fv[0] = tf_fv;
			tf_idf_fv[1] = idf_fv;

			tf_idf_fvs.add(new Instance(tf_idf_fv, inst.getTarget(), 
					inst.getName(), inst.getSource()));
		}

		double threshold = 0.2;
		double damping = 0.1;
		double error = 0.1;

		LexRank lr = new LexRank(tf_idf_fvs, threshold, damping, error);
		lr.rank();
		double[] L = lr.getScore();

		ArrayList<Double> score = new ArrayList<Double>();
		for (double d : L) {
			score.add(d);
		}

		// greedy selection
		int LengthLimit = 100;
		int iterTime = 0;
		Summary currentSummary = new Summary();
		double redundancyThreshold = 0.01;
		Instance inst = tf_idf_fvs.get(getHighestPosition(score));
		currentSummary.add(inst);
		do {
			int position = getHighestPosition(score);
			if (position != -1) {
				Instance candidate = tf_idf_fvs.get(position);
				double redunScore = getRedundancyScore(candidate,
						currentSummary);

				if (redunScore < redundancyThreshold) {
					currentSummary.add(candidate);
					if (currentSummary.length() > LengthLimit) {
						currentSummary.remove(candidate);
						break;
					}
				}
			}

		} while (currentSummary.length() < LengthLimit && (iterTime++) < score.size()/10);
		
		//output
		PrintWriter out = FileOperation.getPrintWriter(new File(outputSummaryDir), corpusName);

		for (Instance sent : currentSummary) {
	     	out.println(sent.getSource());
		}
		out.close();

	}

}
