package edu.pengli.nlp.conference.cikm2012.types;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Sentence;

public class TweetCorpus extends InstanceList implements Serializable {
	
	public TweetCorpus(Iterator<Instance> ii, PipeLine pipeLine) {
		super(pipeLine);
		addThruPipe(ii);
	}
	
	public TweetCorpus(InstanceList users, PipeLine pipeLine) {
		super(pipeLine);
		for (Instance user : users) {
			String text = (String) user.getData();
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
				if(length <= 5) continue;
				Instance inst = new Instance(sent, null, i+"_"+user.getName(), sent);
				ii.add(inst);
			}
			if(ii.size() == 0) continue;
			user.setData(sentsList);
			user.setSource(sentsList);
			sentsList.addThruPipe(ii.iterator());
			this.setDataAlphabet(sentsList.getDataAlphabet());
			add(user);
		}
	
	
	}

}
