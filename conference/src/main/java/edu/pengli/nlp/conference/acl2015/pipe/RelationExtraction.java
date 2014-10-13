package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.HashMap;
import java.util.List;

import scala.collection.Seq;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;
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
		
		HashMap<CoreMap, Seq<edu.knowitall.openie.Instance>> map = 
				new HashMap<CoreMap, Seq<edu.knowitall.openie.Instance>>();

		for (CoreMap sentence : sentences) {		
			Seq<edu.knowitall.openie.Instance> extractions = openIE
					.extract(sentence.toString());
			map.put(sentence, extractions);
		}
		
		instance.setData(map);
		return instance;
	}

}
