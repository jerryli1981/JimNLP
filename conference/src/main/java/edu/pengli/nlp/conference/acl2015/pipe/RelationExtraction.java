package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import scala.collection.Iterator;
import scala.collection.Seq;
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
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class RelationExtraction extends Pipe {

	OpenIE openIE;

	public RelationExtraction() {
		openIE = new OpenIE(new ClearParser(new ClearPostagger(
				new ClearTokenizer(ClearTokenizer.defaultModelUrl()))),
				new ClearSrl(), false);
	}

	public Instance pipe(Instance instance) {

		Annotation document = (Annotation) instance.getData();
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		
		HashMap<CoreMap, ArrayList<Tuple>> map = 
				new HashMap<CoreMap, ArrayList<Tuple>>();

		for (CoreMap sentence : sentences) {		
			Seq<edu.knowitall.openie.Instance> extractions = openIE
					.extract(sentence.toString());
			
			Iterator<edu.knowitall.openie.Instance> iterator = extractions.iterator();
			ArrayList<Tuple> tuples = new ArrayList<Tuple>();
			while (iterator.hasNext()) {
				edu.knowitall.openie.Instance inst = iterator.next();
				double confidence = inst.confidence();
				String arg1 = inst.extr().arg1().text();
				String rel = inst.extr().rel().text();			
				Iterator<Argument> argIter = inst.extr().arg2s().iterator();
				while (argIter.hasNext()) {
					Argument arg2 = argIter.next();
					Tuple t = new Tuple(confidence, arg1, rel, arg2.text());
					tuples.add(t);
				}
			}
			map.put(sentence, tuples);
		}		
		instance.setData(map);
		return instance;
	}
}
