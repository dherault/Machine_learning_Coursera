'use strict';

const isNumeric = require('./utils/isNumeric');
const createIdentityMatrixData = require('./utils/createIdentityMatrixData');

class Matrix {
  
  constructor(array) {
    
    if (arguments.length > 1) console.log('Warning: found more than one argument for new Matrix. Did you forget to wrap your rows in an array?');
    if (!Array.isArray(array) || !array.length) throw new Error('new Matrix: constructor expects an non-empty array.');
    
    this.data = Array.isArray(array[0]) ? array : [array]; // Vector shortcut
    
    const nCol = this.data[0].length;
    
    // Will check column uniformity and if values are numeric
    this.data.forEach((row, i) => {
      if (!Array.isArray(row)) throw new Error(`new Matrix: row ${i + 1} is not an array`);
      if (row.length !== nCol) throw new Error(`new Matrix: inconsistent column number at row ${i + 1}`);
      
      row.forEach((item, j) => {
        
        if (!isNumeric(item)) throw new Error(`new Matrix: found non-numeric value at (${i + 1}, ${j + 1}): ${item}`);
      });
    });
    
    // A malicious user could reassign those variables.
    // This could be prevented with internals + getters
    this.nRow = this.data.length;
    this.nCol = this.data[0].length;
    this.dimension = this.nRow * this.nCol;
    this.isSquare = this.nRow === this.nCol;
    this.isVector = this.nRow === 1 || this.nCol === 1; // We weirdly use the same class for matrixs and vectors
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
    if (!(matrix instanceof Matrix)) throw new Error('Matrix.add: expected arg to be a matrix');
    if (matrix.nRow !== this.nRow || matrix.nCol !== this.nCol) throw new Error('Matrix.add: dimension error');
    
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
    
    if (!(x instanceof Matrix) && !gotScalar) throw new Error('Matrix.multiply: expected arg to be a matrix, a vector or a scalar');
    
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
    if (!(matrix instanceof Matrix)) throw new Error('Matrix.multiplyMatrix: expected arg to be a matrix');
    if (this.nCol !== matrix.nRow) throw new Error('Matrix.multiplyMatrix: dimension error');
    
    const newData = [];
    
    for (let i = 0; i < this.nRow; i++) {
      for (let j = 0; j < matrix.nCol; j++) {
        
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
  
  scalarProduct(vector) {
    if (!(vector instanceof Matrix) || !vector.isVector) throw new Error('Matrix.scalarProduct: expected arg to be a vector');
    if (!this.isVector) throw new Error('Matrix.scalarProduct: current matrix is not a vector');
    if (this.nCol !== vector.nRow) throw new Error('Matrix.scalarProduct: dimension error');
    if (this.nRow !== 1) throw new Error('Matrix.scalarProduct: current vector must be horizontal');
    
    let sum = 0;
    
    for (let i = 0; i < this.nCol; i++) {
      sum += this.data[0][i] * vector.data[i][0];
    }
    
    return sum;
  }
  
  // Returns a deep copy of this.data
  getDataClone() {
    return this.data.map(row => row.slice());
  }
  
  // http://www.math.sciences.univ-nantes.fr/~morame/SYM03/DiagHT/node17.html
  getDeterminant() {
    if (!this.isSquare) throw new Error('Matrix.getDeterminant: current matrix not square');
    
    const size = this.nRow;
    const data = this.getDataClone();
    let det = 1;
    
    for (let j = 0; j < size - 1; j++) {
      
      if (data[j][j] === 0) {
        let firstNonNullPos;
        for (let i = j + 1; i < size; i++) {
          if (data[i][j] !== 0) {
            firstNonNullPos = i;
            break;
          }
        }
        
        if (!firstNonNullPos) return 0;
        
        const swap = data[firstNonNullPos].map(x => -x);
        data[firstNonNullPos] = data[j];
        data[j] = swap;
        
      } else {
        for (let i = j + 1; i < size; i++) {
          const coef = -data[i][j] / data[j][j];
          data[i] = data[i].map((x, pos) => x + coef * data[j][pos]);
        }
      }
      
      det *= data[j][j];
    }
    
    return det * data[size - 1][size - 1];
  }
  
  // https://fr.wikipedia.org/wiki/%C3%89limination_de_Gauss-Jordan#Algorithme
  inverse() {
    
    if (!this.isSquare) throw new Error('Matrix.inverse: current matrix not square');
    if (!this.getDeterminant()) throw new Error('Matrix.inverse: current matrix\'s determinant is 0, cannot inverse');
    
    const size = this.nRow;
    const identityData = createIdentityMatrixData(size);
    // Augmented matrix data:
    const data = this.getDataClone().map((row, k) => row.concat(identityData[k]));
    
    // Gauss-Jordan algorithm
    let r = -1;
    
    for (let j = 0; j < size; j++) {
      
      let k = 0; // data[k][j] is the pivot
      let max = 0;
      
      for (let i = r + 1; i < size; i++) {
        const val = Math.abs(data[i][j]);
        if (val > max) {
          max = val;
          k = i;
        }
      }
      
      const pivot = data[k][j];
      
      if (pivot !== 0) {
        r++;
        const swap = data[k].map((x, pos) => x / pivot);
        data[k] = data[r];
        data[r] = swap;
        
        for (let i = 0; i < size; i++) {
          if (i !== r) {
            const coef = data[i][j];
            data[i] = data[i].map((x, pos) => x - coef * data[r][pos]);
          }
        }
      }
    }
    
    return new Matrix(data.map(row => row.slice(size, 2 * size)));
  }
}

module.exports = Matrix;
