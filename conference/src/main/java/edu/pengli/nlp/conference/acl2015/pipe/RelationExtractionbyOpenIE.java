package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import scala.collection.Iterator;
import scala.collection.Seq;
import edu.knowitall.collection.immutable.Interval;
import edu.knowitall.openie.Argument;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.Instance;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class RelationExtractionbyOpenIE extends Pipe {

	OpenIE openIE;

	public RelationExtractionbyOpenIE() {
		openIE = new OpenIE(new ClearParser(new ClearPostagger(
				new ClearTokenizer(ClearTokenizer.defaultModelUrl()))),
				new ClearSrl(), false);
	}

	public void debug() {

		String yy = "Associated Press writer Martha Raffaele and "
				+ "photographer Carolyn Kaster contributed to this report.";

		Seq<edu.knowitall.openie.Instance> xx = openIE.extract(yy);

		Iterator<edu.knowitall.openie.Instance> iteratorX = xx.iterator();
		while (iteratorX.hasNext()) {
			edu.knowitall.openie.Instance inst = iteratorX.next();
			Seq<Interval> offsets = inst.extr().arg1().offsets();
			Iterator<Interval> ii = offsets.iterator();
			while (ii.hasNext()) {
				Interval in = ii.next();
				int start = in.start();
				int end = in.end();
				System.out.println(yy.substring(start, end));
			}

			Seq<Interval> offsets2 = inst.extr().rel().offsets();
			Iterator<Interval> ii2 = offsets2.iterator();
			while (ii2.hasNext()) {
				Interval in = ii2.next();
				int start = in.start();
				int end = in.end();
				System.out.println(yy.substring(start, end));
			}

			String rel = inst.extr().rel().text();// [is] writer [of]
			if (true) {

				if (rel.matches(".*\\[.*?\\].*")) {
					System.out.println();
					continue;
				}
				System.out.println();
			}
		}

	}

	public Instance pipe(Instance instance) {

		Annotation document = (Annotation) instance.getData();
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		HashMap<CoreMap, ArrayList<Tuple>> map = new HashMap<CoreMap, ArrayList<Tuple>>();

		for (CoreMap sentence : sentences) {

			HashMap<String, CoreLabel> wordLabelMap = new HashMap<String, CoreLabel>();
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				String word = token.get(TextAnnotation.class);
				wordLabelMap.put(word, token);
			}

			Seq<edu.knowitall.openie.Instance> extractions = openIE
					.extract(sentence.toString());

			Iterator<edu.knowitall.openie.Instance> iterator = extractions
					.iterator();
			ArrayList<Tuple> tuples = new ArrayList<Tuple>();
			while (iterator.hasNext()) {
				edu.knowitall.openie.Instance inst = iterator.next();
				int itemSize = 2;
				Iterator<Argument> argiter = inst.extr().arg2s().iterator();
				while (argiter.hasNext()) {
					argiter.next();
					itemSize++;
				}
				double confidence = inst.confidence();
				if (itemSize == 2) {
					continue;
				}
				if (itemSize == 3 || itemSize == 4) {

					String arg1 = inst.extr().arg1().text();
					String rel = inst.extr().rel().text();
					if (rel.matches(".*\\[.*?\\].*")) {
						continue;
					}
					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					while (argIter.hasNext()) {
						String arg2 = argIter.next().text();
						String[] arg1Toks = arg1.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = new edu.pengli.nlp.conference.acl2015.types.Argument();
						for (int i = 0; i < arg1Toks.length; i++) {
							if (wordLabelMap.containsKey(arg1Toks[i])) {
								Arg1.add(wordLabelMap.get(arg1Toks[i]));
							}
						}

						String[] arg2Toks = arg2.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = new edu.pengli.nlp.conference.acl2015.types.Argument();
						for (int i = 0; i < arg2Toks.length; i++) {
							if (wordLabelMap.containsKey(arg2Toks[i])) {
								Arg2.add(wordLabelMap.get(arg2Toks[i]));
							}
						}

						String[] relToks = rel.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Predicate Rel = new edu.pengli.nlp.conference.acl2015.types.Predicate();
						for (int i = 0; i < relToks.length; i++) {
							if (wordLabelMap.containsKey(relToks[i])) {
								Rel.add(wordLabelMap.get(relToks[i]));
							}
						}

						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				} else if (itemSize > 4) {

					String arg1 = inst.extr().arg1().text();
					String rel = inst.extr().rel().text();
					if (rel.matches(".*\\[.*?\\].*")) {
						continue;
					}
					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					ArrayList<Argument> argList = new ArrayList<Argument>();
					while (argIter.hasNext()) {
						Argument arg2 = argIter.next();
						argList.add(arg2);
					}
					String newRel = rel + " " + argList.get(0).text();
					for (int i = 1; i < argList.size(); i++) {

						String[] arg1Toks = arg1.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = new edu.pengli.nlp.conference.acl2015.types.Argument();
						for (int j = 0; j < arg1Toks.length; j++) {
							if (wordLabelMap.containsKey(arg1Toks[j])) {
								Arg1.add(wordLabelMap.get(arg1Toks[j]));
							}
						}

						String[] relToks = newRel.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Predicate Rel = new edu.pengli.nlp.conference.acl2015.types.Predicate();
						for (int j = 0; j < relToks.length; j++) {
							if (wordLabelMap.containsKey(relToks[j])) {
								Rel.add(wordLabelMap.get(relToks[j]));
							}
						}

						String arg2 = argList.get(i).text();
						String[] arg2Toks = arg2.split("\\s|,");
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = new edu.pengli.nlp.conference.acl2015.types.Argument();
						for (int j = 0; j < arg2Toks.length; j++) {
							if (wordLabelMap.containsKey(arg2Toks[j])) {
								Arg2.add(wordLabelMap.get(arg2Toks[j]));
							}
						}

						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				} else {
					System.out.println("Item size is wired");
					System.out.println(inst.toString());
					System.exit(0);

				}

			}
			map.put(sentence, tuples);
		}
		instance.setData(map);
		return instance;
	}

	// for testing
	public static void main(String[] args) {
		RelationExtractionbyOpenIE xx = new RelationExtractionbyOpenIE();
		xx.debug();
	}
}
