package edu.pengli.nlp.conference.cikm2012.io;

import java.io.IOException;
import java.io.ObjectOutputStream;

import edu.pengli.nlp.conference.cikm2012.pipe.CharSequenceExtractParagraph;
import edu.pengli.nlp.platform.algorithms.lda.LDAModel;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.InstanceList;

public class TestRunLDAOnNewsCorpus {

	public static void main(String[] args){

		// Begin by importing documents from text to feature sequences
		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractParagraph());
		pipeLine.addPipe(new CharSequence2TokenSequence());
		pipeLine.addPipe(new TokenSequenceLowercase());
		pipeLine.addPipe(new TokenSequenceRemoveStopwords());
		pipeLine.addPipe(new TokenSequence2FeatureSequence());
		InstanceList instances = new InstanceList(pipeLine);

		// one instance per file
		int topIdx = 1;
		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				"/home/peng/Develop/Workspace/NLP/data/EMNLP2012/Topics/Google/"
						+ String.valueOf(topIdx));
		instances.addThruPipe(fIter);

		double alpha = 10; // 50/numTopics
		double beta = 0.01;
		int numTopics = 5;
		int numIters = 100;

		LDAModel model = new LDAModel(numTopics, alpha, beta, numIters);

		model.initEstimate(instances);

		model.estimate();

		int topK = 10; // for each topic output top K words
		model.outputModel(topK);

		String outputDir = "/home/peng/Develop/Workspace/NLP/data/EMNLP2012/";
		String outputName = "newsTopicsGoogle";
		model.writeModel(outputDir, outputName);

		System.out.println("done");

	}
}
