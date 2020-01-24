import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
/*
  Generated class for the AuthServiceProvider provider.

  See https://angular.io/guide/dependency-injection for more info on providers
  and Angular DI.
*/
@Injectable()
export class AuthServiceProvider {

  constructor(public http: HttpClient, private serv: AuthServiceProvider) {
    console.log('Hello AuthServiceProvider Provider');
  }
  getNumber(img:any): Observable<any> {
    let url = 'http://10.40.46.25:5000/';
    let apiURL = url;
    return this.http.post<any>(apiURL, img);
  }
}
