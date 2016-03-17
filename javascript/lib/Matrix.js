'use strict';

const isNumeric = require('./utils/isNumeric');

class Matrix {
  
  constructor(array) {
    
    if (arguments.length > 1) console.log('Warning: found more than one argument for new Matrix. Did you forget to wrap your rows in an array?');
    if (!Array.isArray(array) || !array.length) throw new Error('new Matrix: constructor expects an non-empty array.');
    
    this.data = Array.isArray(array[0]) ? array : [array]; // Vector shortcut
    
    const nCol = this.data[0].length;
    
    // Will check column uniformity and if values are numeric
    this.data.forEach((row, i) => {
      
      if (row.length !== nCol) throw new Error(`new Matrix: inconsistent column number at row ${i}`);
      
      row.forEach((item, j) => {
        
        if (!isNumeric(item)) throw new Error(`new Matrix: found non-numeric value at (${i}, ${j}): ${item}`);
      });
    });
    
    this.nRow = this.data.length;
    this.nCol = this.data[0].length;
    this.dimension = this.nRow * this.nCol;
  }
  
  transpose() {
    const newData = [];
    
    for (let i = 0; i < this.nRow; i++) {
      for (let j = 0; j < this.nCol; j++) {
        if (!newData[j]) newData[j] = [];
        newData[j].push(this.data[i][j]);
      }
    }
    
    return new Matrix(newData);
  }
  
  add(matrix) {
    if (!matrix instanceof Matrix) throw new Error('Matrix.add: expected arg to be a matrix');
    if (matrix.nRow !== this.nRow || matrix.nCol !== this.nCol) throw new Error('Matrix.add: given matrix does not match current matrix\'s dimensions');
    
    const newData = [];
    for (let i = 0; i < this.nRow; i++) {
      for (let j = 0; j < this.nCol; j++) {
        if (!newData[i]) newData[i] = [];
        newData[i].push(this.data[i][j] + matrix.data[i][j]);
      }
    }
    
    return new Matrix(newData);
  }
  
  multiply(x) {
    const gotScalar = isNumeric(x);
    
    if (!x instanceof Matrix && !gotScalar) throw new Error('Matrix.multiply: expected arg to be a matrix or a number');
    
    return gotScalar ? this.multiplyScalar(x) : this.multiplyMatrix(x);
  }
  
  multiplyScalar(x) {
    if (!isNumeric(x)) throw new Error('Matrix.multiplyScalar: expected arg to be a number');
    
    const newData = [];
    
    for (let i = 0; i < this.nRow; i++) {
      for (let j = 0; j < this.nCol; j++) {
        if (!newData[i]) newData[i] = [];
        newData[i].push(x * this.data[i][j]);
      }
    }
    
    return new Matrix(newData);
  }
  
  multiplyMatrix(matrix) {
    if (!matrix instanceof Matrix) throw new Error('Matrix.multiplyMatrix: expected arg to be a matrix');
    if (this.nRow !== matrix.nCol || this.nCol !== matrix.nRow) throw new Error('Matrix.multiplyMatrix: given matrix does not match current matrix\'s dimensions');
    
    const newData = [];
    
    for (let i = 0; i < this.nRow; i++) {
      for (let j = 0; j < this.nRow; j++) {
        
        if (!newData[i]) newData[i] = [];
        let sum = 0;
        
        for (let k = 0; k < this.nCol; k++) {
          sum += this.data[i][k] * matrix.data[k][j];
        }
        
        newData[i].push(sum);
      }
    }
    
    return new Matrix(newData);
  }
}

module.exports = Matrix;
