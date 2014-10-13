package edu.pengli.nlp.platform.types;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class Instance implements Serializable{

	protected Object data;
	protected Object target;
	protected Object name;
	protected Object source;

	public Instance(Object data, Object target, Object name) {
		this.data = data;
		this.target = target;
		this.name = name;

	}
	
	public Instance(Object data, Object target, Object name, Object source) {
		this.data = data;
		this.target = target;
		this.name = name;
		this.source = source;

	}
	
	public Alphabet getDataAlphabet() {
		if (data instanceof AlphabetCarrying)
			return ((AlphabetCarrying)data).getAlphabet();
		else
			return null;
	}

	public Object getData() {
		return data;
	}

	public void setData(Object d) {
		data = d;
	}

	public Object getTarget() {
		return target;
	}

	public void setTarget(Object target) {
		this.target = target;
	}

	public Object getName() {
		return name;
	}

	public void setName(Object name) {
		this.name = name;
	}
	
	public Object getSource(){
		return source;
	}
	
	public void setSource(Object source){
		this.source = source;
	}
	
	// Serialization of Instance

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(data);
		out.writeObject(target);
		out.writeObject(name);
		out.writeObject(source);
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		data = in.readObject();
		target = in.readObject();
		name = in.readObject();
		source = in.readObject();
	}

}
