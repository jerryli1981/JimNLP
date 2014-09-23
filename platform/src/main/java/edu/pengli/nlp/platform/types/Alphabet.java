package edu.pengli.nlp.platform.types;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

public class Alphabet implements Serializable{

	HashMap<Object, Integer> map;

	ArrayList entries;

	boolean growthStopped = false;
	
	public Alphabet(){
		this(10);
	}

	public Alphabet(int capacity) {
		this.map = new HashMap<Object, Integer>(capacity);
		this.entries = new ArrayList(capacity);
	}

	public int lookupIndex(Object entry, boolean addIfNotPresent) {
		if (entry == null)
			throw new IllegalArgumentException(
					"Can't lookup \"null\" in an Alphabet.");

		int retIndex = -1;
		if (map.containsKey(entry)) {
			retIndex = map.get(entry);
		} else if (!growthStopped && addIfNotPresent) {
			retIndex = entries.size();
			map.put(entry, retIndex);
			entries.add(entry);
		}
		return retIndex;
	}

	public int lookupIndex(Object entry) {
		return lookupIndex(entry, true);
	}

	public Object lookupObject(int index) {
		return entries.get(index);
	}
	
	public Object[] toArray () {
		return entries.toArray();
	}

	public boolean contains(Object entry) {
		return map.containsKey(entry);
	}

	public int size() {
		return entries.size();
	}

	public void stopGrowth() {
		growthStopped = true;
	}

	public void startGrowth() {
		growthStopped = false;
	}

	public boolean growthStopped() {
		return growthStopped;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < entries.size(); i++) {
			sb.append(entries.get(i).toString());
			sb.append('\n');
		}
		return sb.toString();
	}
}
