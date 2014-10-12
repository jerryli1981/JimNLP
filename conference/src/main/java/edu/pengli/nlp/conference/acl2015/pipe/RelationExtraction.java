package edu.pengli.nlp.conference.acl2015.pipe;

import scala.collection.Iterator;
import scala.collection.Seq;
import edu.knowitall.openie.Argument;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.Instance;

public class RelationExtraction extends Pipe{
	
	 OpenIE openIE;

	public RelationExtraction() {
	     openIE = new OpenIE(new ClearParser(new ClearPostagger(
	                new ClearTokenizer(ClearTokenizer.defaultModelUrl()))),
	                new ClearSrl(), false);
	}

	public Instance pipe(Instance instance) {
		
		String text = (String) instance.getData();
        Seq<edu.knowitall.openie.Instance> extractions = 
        		openIE.extract(text);
        Iterator<edu.knowitall.openie.Instance> iterator = extractions.iterator();
        instance.setSource(text);
        instance.setData(iterator);
/*        while (iterator.hasNext()) {
        	edu.knowitall.openie.Instance inst = iterator.next();
            StringBuilder sb = new StringBuilder();
            sb.append(inst.confidence())
                .append('\t')
                .append(inst.extr().arg1().text())
                .append('\t')
                .append(inst.extr().rel().text())
                .append('\t');

            Iterator<Argument> argIter = inst.extr().arg2s().iterator();
            while (argIter.hasNext()) {
                Argument arg = argIter.next();
                sb.append(arg.text()).append("; ");
            }

            System.out.println(sb.toString());
        }*/

		return instance;
	}

}
