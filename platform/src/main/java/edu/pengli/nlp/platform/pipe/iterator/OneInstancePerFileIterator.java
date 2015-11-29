package edu.pengli.nlp.platform.pipe.iterator;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;

import edu.pengli.nlp.platform.types.Instance;

public class OneInstancePerFileIterator implements Iterator<Instance> {
	
	protected Iterator<File> fileIterator;
	
	public OneInstancePerFileIterator(Iterator<File> fileIterator){
		this.fileIterator = fileIterator;
	}
	public OneInstancePerFileIterator(String dir){
	
		File d = new File(dir);
		File[] fl = d.listFiles();
		ArrayList<File> fs = new ArrayList<File>();
		for(int i=0; i<fl.length; i++){
			fs.add(fl[i]);
		}
		fileIterator = fs.iterator();
	}


	public boolean hasNext() {
		// TODO Auto-generated method stub
		return fileIterator.hasNext();
	}


	public Instance next() {
		File f = fileIterator.next();
		String name = f.getName();
		return new Instance(f, null, name);
	}


	public void remove() {
		throw new IllegalStateException("Not supported");
	}

}
