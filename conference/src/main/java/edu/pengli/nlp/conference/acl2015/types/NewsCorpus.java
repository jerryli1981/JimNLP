package edu.pengli.nlp.conference.acl2015.types;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;

public class NewsCorpus extends InstanceList implements Serializable {

	public NewsCorpus(Iterator<Instance> ii, PipeLine pipeLine) {
		super(pipeLine);
		addThruPipe(ii);
	}

	public NewsCorpus(InstanceList docs, PipeLine pipeLine) {
		super(pipeLine);
		for (Instance doc : docs) {
			String text = (String) doc.getData();
			String[] sents = text.split("\\n");
			InstanceList sentsList = new InstanceList(pipeLine);
			ArrayList<Instance> ii = new ArrayList<Instance>();
			for (int i = 0; i < sents.length; i++) {
				String sent = sents[i];
				String[] toks = sent.split(" ");
				int length = 0;
				for(int j=0; j<toks.length; j++){
					String tok = toks[j];
					if(tok.matches("[a-zA-Z_0-9-]+")) length++;
				}
				if(length <= 10) continue;
				Instance inst = new Instance(sent, null, i + "_"
						+ doc.getName(), sent);
				ii.add(inst);
			}
			if(ii.size() == 0) continue;
			sentsList.addThruPipe(ii.iterator());
			this.setDataAlphabet(sentsList.getDataAlphabet());
			doc.setData(sentsList);
			doc.setSource(sentsList);
			add(doc);
		}

	}
}

