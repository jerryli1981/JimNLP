package edu.pengli.nlp.platform.types;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import edu.pengli.nlp.platform.pipe.Pipe;

public class InstanceList extends ArrayList<Instance> implements Serializable {

	Pipe pipe;

	Alphabet dataAlphabet;
	
	public InstanceList(Pipe pipe) {
		this.pipe = pipe;
	}

	public boolean add(Instance instance) {
		if (dataAlphabet == null)
			dataAlphabet = instance.getDataAlphabet();

		return super.add(instance);
	}

	public void addThruPipe(Iterator<Instance> ii) {
		Iterator<Instance> pipedInstanceIterator = pipe.newIteratorFrom(ii);
		while (pipedInstanceIterator.hasNext()) {
			Instance inst = pipedInstanceIterator.next();
			add(inst);
		}
	}

	public void setDataAlphabet(Alphabet dataAlphabet) {
		this.dataAlphabet = dataAlphabet;
	}

	public Alphabet getDataAlphabet() {
		if (dataAlphabet == null && pipe != null) {
			dataAlphabet = pipe.getDataAlphabet();
		}
		return dataAlphabet;
	}

	public void writeObject(ObjectOutputStream out) throws IOException {
			out.writeObject(this.size());
			for(Instance inst : this){
				inst.writeObject(out);
			}	
	}

	public void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
			int size = (Integer)in.readObject();
		    for(int i=0; i<size; i++){
		    	Instance inst = new Instance(null, null, null, null);
		    	inst.readObject(in);
		    	add(inst);
		    }   
	}

	public Pipe getPipe() {
		return pipe;
	}

}
