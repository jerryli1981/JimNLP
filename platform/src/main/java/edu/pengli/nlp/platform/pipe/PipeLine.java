package edu.pengli.nlp.platform.pipe;

import java.util.ArrayList;
import java.util.Iterator;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Instance;

public class PipeLine extends Pipe {

	ArrayList<Pipe> pipeLine;

	public PipeLine() {
		pipeLine = new ArrayList<Pipe>();
	}
	
	public void addPipe(Pipe p){
		pipeLine.add(p);
	}
	
	public Pipe getPipe(int idx){
		return pipeLine.get(idx);
	}
		
	@Override
	public Iterator<Instance> newIteratorFrom(Iterator<Instance> source){
		Iterator<Instance> ret = pipeLine.get(0).newIteratorFrom(source);
		for(int i=1; i<pipeLine.size(); i++)
			ret = pipeLine.get(i).newIteratorFrom(ret);
		return ret;
	}
	
	@Override
	protected Instance pipe(Instance inst) {
		throw new UnsupportedOperationException("Should not call this method");
	}

}
